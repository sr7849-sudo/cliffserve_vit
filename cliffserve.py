#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import yaml
from PIL import Image
from timm.data import create_transform, resolve_model_data_config
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
POMT_ROOT = ROOT / "third_party" / "PruneOneMoreToken"


# ============================================================
# Utilities
# ============================================================

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mkdir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def add_pomt_path():
    if not (POMT_ROOT / "pomt" / "timm_patch.py").exists():
        raise RuntimeError(
            f"POMT repository not found at {POMT_ROOT}\n"
            "Clone https://github.com/nickjeliopoulos/PruneOneMoreToken first."
        )

    if str(POMT_ROOT) not in sys.path:
        sys.path.insert(0, str(POMT_ROOT))


def pomt_commit():
    try:
        return subprocess.check_output(
            ["git", "-C", str(POMT_ROOT), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def hardware_provenance(device):
    result = {
        "torch": torch.__version__,
        "timm": timm.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "pomt_commit": pomt_commit(),
    }

    if torch.cuda.is_available():
        dev = torch.device(device)
        result["gpu"] = torch.cuda.get_device_name(dev)
        result["gpu_capability"] = torch.cuda.get_device_capability(dev)

    try:
        result["nvidia_smi"] = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,power.limit",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
    except Exception:
        pass

    return result


# ============================================================
# ImageNetV2
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".jfif",
}


class ImageNetV2NumericFolders(Dataset):
    """
    ImageNetV2 matched-frequency is stored as:

        root/0/...
        root/1/...
        ...
        root/999/...

    We explicitly interpret the folder name as the ImageNet class ID.

    This is safer than relying on ImageFolder's lexical directory ordering.
    """

    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform

        if not self.root.exists():
            raise FileNotFoundError(self.root)

        samples = []

        class_dirs = [
            p
            for p in self.root.iterdir()
            if p.is_dir() and p.name.isdigit()
        ]

        class_dirs = sorted(
            class_dirs,
            key=lambda x: int(x.name),
        )

        for class_dir in class_dirs:
            target = int(class_dir.name)

            for image_path in sorted(class_dir.iterdir()):
                if (
                    image_path.is_file()
                    and image_path.suffix.lower() in IMAGE_EXTENSIONS
                ):
                    samples.append(
                        (image_path, target)
                    )

        if not samples:
            raise RuntimeError(
                f"No ImageNetV2 images found under {root}"
            )

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]

        with Image.open(path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, target, str(path)


# ============================================================
# Models
# ============================================================

def create_timm_model(model_name, device):
    model = timm.create_model(
        model_name,
        pretrained=True,
    )

    model.eval()
    model.to(device)

    return model


def get_transform(model):
    data_config = resolve_model_data_config(model)

    return create_transform(
        **data_config,
        is_training=False,
    )


def patch_pomt(model, prune_layer):
    add_pomt_path()

    from argparse import Namespace
    from pomt.timm_patch import timm_apply_pomt_patch

    args = Namespace(
        pomt_prune_layer_index=int(prune_layer),
        pomt_R=1,
    )

    model = timm_apply_pomt_patch(
        args,
        model,
    )

    model.eval()

    return model


def set_pomt_r(model, r, prune_layer):
    if prune_layer < 0 or prune_layer >= len(model.blocks):
        raise ValueError(
            f"prune_layer={prune_layer}, "
            f"but model has {len(model.blocks)} blocks"
        )

    schedule = [0] * len(model.blocks)
    schedule[prune_layer] = int(r)

    #
    # POMT's patched model copies this vector during every forward().
    #
    model.r = schedule


def make_models(model_name, device, prune_layer):
    print("Loading true dense model...")

    dense = create_timm_model(
        model_name,
        device,
    )

    transform = get_transform(dense)

    print("Loading second model for dynamic POMT...")

    pomt = create_timm_model(
        model_name,
        device,
    )

    pomt = patch_pomt(
        pomt,
        prune_layer,
    )

    set_pomt_r(
        pomt,
        1,
        prune_layer,
    )

    return dense, pomt, transform


def num_patch_tokens(model):
    return int(
        model.patch_embed.num_patches
    )


def sequence_tokens_after_r(model, r):
    prefix = int(
        getattr(
            model,
            "num_prefix_tokens",
            1,
        )
    )

    return (
        num_patch_tokens(model)
        + prefix
        - int(r)
    )


def content_tokens_after_r(model, r):
    return (
        num_patch_tokens(model)
        - int(r)
    )


# ============================================================
# Sanity
# ============================================================

@torch.inference_mode()
def run_sanity(config):
    cfg = load_config(config)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "torch.cuda.is_available() is False"
        )

    device = cfg["device"]

    dense, pomt, _ = make_models(
        cfg["model"],
        device,
        cfg["prune_layer"],
    )

    x = torch.randn(
        2,
        3,
        224,
        224,
        device=device,
    )

    dense_output = dense(x)

    set_pomt_r(
        pomt,
        64,
        cfg["prune_layer"],
    )

    pomt_output = pomt(x)

    print()
    print("GPU:", torch.cuda.get_device_name(0))
    print("Dense output:", tuple(dense_output.shape))
    print("POMT output:", tuple(pomt_output.shape))
    print("POMT commit:", pomt_commit())
    print()
    print("SANITY CHECK PASSED")


# ============================================================
# CUDA profiling
# ============================================================

@torch.inference_mode()
def measure_cuda_latency(
    model,
    batch_size,
    device,
    warmup,
    repeats,
):
    x = torch.randn(
        batch_size,
        3,
        224,
        224,
        device=device,
        dtype=torch.float32,
    )

    for _ in range(warmup):
        model(x)

    torch.cuda.synchronize()

    samples = np.empty(
        repeats,
        dtype=np.float64,
    )

    start = torch.cuda.Event(
        enable_timing=True
    )

    end = torch.cuda.Event(
        enable_timing=True
    )

    for i in range(repeats):
        start.record()

        model(x)

        end.record()

        torch.cuda.synchronize()

        samples[i] = float(
            start.elapsed_time(end)
        )

    return samples


def profile_latency(
    cfg,
    dense,
    pomt,
    outdir,
):
    device = cfg["device"]

    batch_sizes = cfg["batch_sizes"]
    r_values = cfg["r_values"]

    rows = []
    raw = {}

    print()
    print("=" * 70)
    print("GPU LATENCY PROFILING")
    print("=" * 70)

    for batch_size in batch_sizes:

        for r in r_values:

            if r == 0:
                #
                # IMPORTANT:
                # use the genuinely unpatched model.
                #
                model = dense

            else:
                set_pomt_r(
                    pomt,
                    r,
                    cfg["prune_layer"],
                )

                model = pomt

            print(
                f"B={batch_size:>2} "
                f"R={r:>3} "
                f"content_tokens="
                f"{content_tokens_after_r(dense, r)}"
            )

            samples = measure_cuda_latency(
                model=model,
                batch_size=batch_size,
                device=device,
                warmup=cfg["latency_warmup"],
                repeats=cfg["latency_repeats"],
            )

            key = (
                f"b{batch_size}_r{r}"
            )

            raw[key] = samples

            median = float(
                np.median(samples)
            )

            p95 = float(
                np.percentile(
                    samples,
                    95,
                )
            )

            rows.append(
                {
                    "model": cfg["model"],
                    "batch_size": batch_size,
                    "r": r,

                    "sequence_tokens":
                        sequence_tokens_after_r(
                            dense,
                            r,
                        ),

                    "content_tokens":
                        content_tokens_after_r(
                            dense,
                            r,
                        ),

                    "latency_mean_ms":
                        float(np.mean(samples)),

                    "latency_median_ms":
                        median,

                    "latency_p90_ms":
                        float(
                            np.percentile(
                                samples,
                                90,
                            )
                        ),

                    "latency_p95_ms":
                        p95,

                    "latency_p99_ms":
                        float(
                            np.percentile(
                                samples,
                                99,
                            )
                        ),

                    "latency_std_ms":
                        float(
                            np.std(
                                samples,
                                ddof=1,
                            )
                        ),

                    "throughput_median_qps":
                        float(
                            1000.0
                            * batch_size
                            / median
                        ),

                    "throughput_p95_qps":
                        float(
                            1000.0
                            * batch_size
                            / p95
                        ),
                }
            )

    df = pd.DataFrame(rows)

    df = df.sort_values(
        [
            "batch_size",
            "r",
        ]
    ).reset_index(drop=True)

    df.to_csv(
        outdir / "latency.csv",
        index=False,
    )

    np.savez_compressed(
        outdir / "latency_samples.npz",
        **raw,
    )

    return df


# ============================================================
# Accuracy
# ============================================================

@torch.inference_mode()
def evaluate_accuracy(
    cfg,
    dense,
    pomt,
    dataset,
    outdir,
):
    n = min(
        cfg["accuracy_samples"],
        len(dataset),
    )

    #
    # Same examples for every B,R.
    #
    subset = Subset(
        dataset,
        list(range(n)),
    )

    rows = []

    print()
    print("=" * 70)
    print("ACCURACY SURFACE")
    print("=" * 70)

    for batch_size in cfg["batch_sizes"]:

        dataloader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )

        for r in cfg["r_values"]:

            if r == 0:
                model = dense

            else:
                set_pomt_r(
                    pomt,
                    r,
                    cfg["prune_layer"],
                )

                model = pomt

            correct = 0
            total = 0

            description = (
                f"B={batch_size} "
                f"R={r}"
            )

            for (
                images,
                targets,
                _,
            ) in tqdm(
                dataloader,
                desc=description,
                leave=False,
            ):

                images = images.to(
                    cfg["device"],
                    non_blocking=True,
                )

                targets = targets.to(
                    cfg["device"],
                    non_blocking=True,
                )

                logits = model(images)

                predictions = (
                    logits.argmax(
                        dim=1
                    )
                )

                correct += int(
                    (
                        predictions
                        == targets
                    )
                    .sum()
                    .item()
                )

                total += int(
                    targets.numel()
                )

            accuracy = (
                100.0
                * correct
                / total
            )

            rows.append(
                {
                    "model":
                        cfg["model"],

                    "batch_size":
                        batch_size,

                    "r":
                        r,

                    "accuracy_pct":
                        accuracy,

                    "correct":
                        correct,

                    "num_samples":
                        total,
                }
            )

            print(
                f"B={batch_size:>2} "
                f"R={r:>3} "
                f"accuracy={accuracy:.3f}%"
            )

    df = pd.DataFrame(rows)

    df.to_csv(
        outdir / "accuracy.csv",
        index=False,
    )

    return df


# ============================================================
# Surface / Pareto / hardware cliffs
# ============================================================

def mark_pareto(surface):
    surface = surface.copy()

    surface["pareto"] = False

    for batch_size, group in surface.groupby(
        "batch_size"
    ):

        indices = list(group.index)

        for i in indices:

            ai = float(
                surface.at[
                    i,
                    "accuracy_pct",
                ]
            )

            li = float(
                surface.at[
                    i,
                    "latency_p95_ms",
                ]
            )

            dominated = False

            for j in indices:

                if i == j:
                    continue

                aj = float(
                    surface.at[
                        j,
                        "accuracy_pct",
                    ]
                )

                lj = float(
                    surface.at[
                        j,
                        "latency_p95_ms",
                    ]
                )

                if (
                    aj >= ai
                    and lj <= li
                    and (
                        aj > ai
                        or lj < li
                    )
                ):
                    dominated = True
                    break

            if not dominated:
                surface.at[
                    i,
                    "pareto",
                ] = True

    return surface


def build_surface(
    latency,
    accuracy,
):
    surface = latency.merge(
        accuracy,
        on=[
            "model",
            "batch_size",
            "r",
        ],
        validate="one_to_one",
    )

    dense_accuracy = (
        surface[
            surface["r"] == 0
        ][
            [
                "batch_size",
                "accuracy_pct",
            ]
        ]
        .rename(
            columns={
                "accuracy_pct":
                    "dense_accuracy_pct",
            }
        )
    )

    surface = surface.merge(
        dense_accuracy,
        on="batch_size",
        how="left",
    )

    surface[
        "accuracy_drop_pp"
    ] = (
        surface[
            "dense_accuracy_pct"
        ]
        - surface[
            "accuracy_pct"
        ]
    )

    surface[
        "speedup_vs_dense_p95"
    ] = np.nan

    for batch_size, group in surface.groupby(
        "batch_size"
    ):

        dense_latency = float(
            group[
                group["r"] == 0
            ][
                "latency_p95_ms"
            ]
            .iloc[0]
        )

        surface.loc[
            group.index,
            "speedup_vs_dense_p95",
        ] = (
            dense_latency
            / group[
                "latency_p95_ms"
            ]
        )

    return mark_pareto(
        surface
    )


def build_cliff_frontier(
    surface,
    epsilon_pp,
    min_relative_gain,
    min_absolute_gain_ms,
):
    pieces = []

    for batch_size, group in surface.groupby(
        "batch_size"
    ):

        feasible = group[
            group[
                "accuracy_drop_pp"
            ]
            <= epsilon_pp
        ].copy()

        feasible = feasible.sort_values(
            "r"
        )

        if feasible.empty:
            continue

        retained = []

        previous_latency = None

        for _, row in feasible.iterrows():

            current = float(
                row[
                    "latency_p95_ms"
                ]
            )

            if previous_latency is None:

                retained.append(
                    row
                )

                previous_latency = (
                    current
                )

                continue

            absolute_gain = (
                previous_latency
                - current
            )

            relative_gain = (
                absolute_gain
                / max(
                    previous_latency,
                    1e-12,
                )
            )

            #
            # Only sacrifice more tokens when
            # hardware actually gives something
            # useful in return.
            #
            if (
                absolute_gain
                >= min_absolute_gain_ms
                or relative_gain
                >= min_relative_gain
            ):

                retained.append(
                    row
                )

                previous_latency = (
                    current
                )

        #
        # Always retain globally fastest
        # quality-feasible point.
        #
        fastest = feasible.loc[
            feasible[
                "latency_p95_ms"
            ]
            .idxmin()
        ]

        if not any(
            int(x["r"])
            == int(fastest["r"])
            for x in retained
        ):
            retained.append(
                fastest
            )

        result = pd.DataFrame(
            retained
        )

        result = (
            result
            .drop_duplicates(
                subset=[
                    "batch_size",
                    "r",
                ]
            )
        )

        pieces.append(
            result
        )

    if not pieces:
        return surface.iloc[0:0]

    return (
        pd.concat(
            pieces,
            ignore_index=True,
        )
        .sort_values(
            [
                "batch_size",
                "r",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# Profile command
# ============================================================

def run_profile(config, dataset_root, outdir):
    cfg = load_config(config)

    seed_all(84)

    output = mkdir(outdir)

    provenance = hardware_provenance(
        cfg["device"]
    )

    provenance["config"] = cfg

    with open(
        output / "provenance.json",
        "w",
    ) as f:
        json.dump(
            provenance,
            f,
            indent=2,
            default=str,
        )

    dense, pomt, transform = make_models(
        cfg["model"],
        cfg["device"],
        cfg["prune_layer"],
    )

    dataset = ImageNetV2NumericFolders(
        dataset_root,
        transform=transform,
    )

    print(
        "Dataset samples:",
        len(dataset),
    )

    latency = profile_latency(
        cfg,
        dense,
        pomt,
        output,
    )

    accuracy = evaluate_accuracy(
        cfg,
        dense,
        pomt,
        dataset,
        output,
    )

    surface = build_surface(
        latency,
        accuracy,
    )

    surface.to_csv(
        output / "surface.csv",
        index=False,
    )

    cliff = build_cliff_frontier(
        surface=surface,

        epsilon_pp=
            cfg["epsilon_pp"],

        min_relative_gain=
            cfg[
                "cliff_min_relative_gain"
            ],

        min_absolute_gain_ms=
            cfg[
                "cliff_min_absolute_gain_ms"
            ],
    )

    cliff.to_csv(
        output
        / "cliff_frontier.csv",

        index=False,
    )

    print()
    print("=" * 70)
    print("CLIFF FRONTIER")
    print("=" * 70)

    print(
        cliff[
            [
                "batch_size",
                "r",
                "content_tokens",
                "latency_p95_ms",
                "accuracy_pct",
                "accuracy_drop_pp",
                "speedup_vs_dense_p95",
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# Scheduling
# ============================================================

@dataclass(frozen=True)
class Request:
    request_id: int
    arrival_ms: float
    deadline_ms: float
    sample_index: int


@dataclass
class Decision:
    batch_size: int
    r: int
    predicted_service_ms: float
    predicted_accuracy_pct: float
    scheduler_overhead_us: float


class SurfaceIndex:

    def __init__(
        self,
        surface,
        epsilon_pp,
        cliff_candidates=None,
    ):

        self.surface = surface.copy()

        self.epsilon_pp = (
            float(
                epsilon_pp
            )
        )

        self.by_key = {
            (
                int(row.batch_size),
                int(row.r),
            ): row

            for row
            in self.surface.itertuples()
        }

        self.batch_sizes = sorted(
            set(
                int(x)
                for x
                in self.surface[
                    "batch_size"
                ]
            )
        )

        if cliff_candidates is None:
            self.cliff_keys = set(
                self.by_key.keys()
            )
        else:
            self.cliff_keys = {
                (
                    int(row.batch_size),
                    int(row.r),
                )
                for row
                in cliff_candidates.itertuples()
            }

    def dense(self, batch_size):
        return self.by_key[
            (
                int(batch_size),
                0,
            )
        ]

    def feasible(
        self,
        batch_size,
        cliff_only=False,
    ):
        result = []

        for row in self.surface.itertuples():

            if (
                int(row.batch_size)
                != int(batch_size)
            ):
                continue

            if (
                float(
                    row.accuracy_drop_pp
                )
                > self.epsilon_pp
            ):
                continue

            key = (
                int(row.batch_size),
                int(row.r),
            )

            if (
                cliff_only
                and key
                not in self.cliff_keys
            ):
                continue

            result.append(
                row
            )

        return result


def remaining_slack(
    queue,
    batch_size,
    now,
):
    return min(
        req.deadline_ms
        - now

        for req
        in queue[
            :batch_size
        ]
    )


class BaseScheduler:

    def __init__(
        self,
        index,
        max_batch,
    ):
        self.index = index
        self.max_batch = max_batch

    def candidates(self, queue):
        return [
            b
            for b
            in self.index.batch_sizes

            if (
                b <= len(queue)
                and b
                <= self.max_batch
            )
        ]

    def make_decision(
        self,
        start_ns,
        batch_size,
        r,
        row,
    ):
        return Decision(
            batch_size=int(
                batch_size
            ),

            r=int(r),

            predicted_service_ms=float(
                row.latency_p95_ms
            ),

            predicted_accuracy_pct=float(
                row.accuracy_pct
            ),

            scheduler_overhead_us=(
                perf_counter_ns()
                - start_ns
            )
            / 1000.0,
        )


class DenseScheduler(BaseScheduler):

    def decide(
        self,
        queue,
        now,
    ):
        start = perf_counter_ns()

        candidates = (
            self.candidates(
                queue
            )
        )

        feasible = []

        for b in candidates:

            row = (
                self.index.dense(
                    b
                )
            )

            slack = remaining_slack(
                queue,
                b,
                now,
            )

            if (
                float(
                    row.latency_p95_ms
                )
                <= slack
            ):

                feasible.append(
                    (
                        b,
                        row,
                    )
                )

        if feasible:

            b, row = max(
                feasible,
                key=lambda x:
                    x[0],
            )

        else:

            b = min(
                candidates
            )

            row = (
                self.index.dense(
                    b
                )
            )

        return self.make_decision(
            start,
            b,
            0,
            row,
        )


def choose_global_static_r(
    index,
):
    #
    # Require static R to satisfy accuracy
    # budget at *all* batch sizes.
    #

    common = None

    for b in index.batch_sizes:

        rs = {
            int(row.r)
            for row
            in index.feasible(b)
        }

        if common is None:
            common = rs
        else:
            common &= rs

    if not common:
        return 0

    best_r = 0
    best_score = -math.inf

    for r in common:

        speedups = []

        for b in index.batch_sizes:

            dense = (
                index.dense(b)
            )

            row = index.by_key[
                (
                    b,
                    r,
                )
            ]

            speedups.append(
                float(
                    dense.latency_p95_ms
                )
                / float(
                    row.latency_p95_ms
                )
            )

        score = float(
            np.mean(
                speedups
            )
        )

        if score > best_score:

            best_score = score
            best_r = r

    return best_r


class StaticPOMTScheduler(
    BaseScheduler
):

    def __init__(
        self,
        index,
        max_batch,
    ):

        super().__init__(
            index,
            max_batch,
        )

        self.r = (
            choose_global_static_r(
                index
            )
        )

    def decide(
        self,
        queue,
        now,
    ):

        start = perf_counter_ns()

        candidates = (
            self.candidates(
                queue
            )
        )

        feasible = []

        for b in candidates:

            key = (
                b,
                self.r,
            )

            if key not in self.index.by_key:
                continue

            row = self.index.by_key[
                key
            ]

            if (
                float(
                    row.accuracy_drop_pp
                )
                > self.index.epsilon_pp
            ):
                continue

            slack = remaining_slack(
                queue,
                b,
                now,
            )

            if (
                float(
                    row.latency_p95_ms
                )
                <= slack
            ):
                feasible.append(
                    (
                        b,
                        row,
                    )
                )

        if feasible:

            b, row = max(
                feasible,
                key=lambda x:
                    x[0],
            )

            r = self.r

        else:

            b = min(
                candidates
            )

            row = (
                self.index.dense(
                    b
                )
            )

            r = 0

        return self.make_decision(
            start,
            b,
            r,
            row,
        )


class DecoupledScheduler(
    BaseScheduler
):

    def decide(
        self,
        queue,
        now,
    ):

        start = perf_counter_ns()

        candidates = (
            self.candidates(
                queue
            )
        )

        #
        # Stage 1:
        # select B using dense inference
        # only.
        #
        valid_batches = []

        for b in candidates:

            dense = (
                self.index.dense(
                    b
                )
            )

            slack = remaining_slack(
                queue,
                b,
                now,
            )

            if (
                float(
                    dense.latency_p95_ms
                )
                <= slack
            ):
                valid_batches.append(
                    b
                )

        if valid_batches:

            b = max(
                valid_batches
            )

        else:

            b = min(
                candidates
            )

        #
        # Stage 2:
        # only after B is fixed,
        # choose R.
        #
        slack = remaining_slack(
            queue,
            b,
            now,
        )

        feasible = [
            row
            for row
            in self.index.feasible(
                b
            )

            if (
                float(
                    row.latency_p95_ms
                )
                <= slack
            )
        ]

        if not feasible:

            row = (
                self.index.dense(
                    b
                )
            )

            r = 0

        else:

            #
            # Since B is fixed,
            # maximum throughput =
            # minimum latency.
            #
            row = min(
                feasible,
                key=lambda x:
                    float(
                        x.latency_p95_ms
                    ),
            )

            r = int(
                row.r
            )

        return self.make_decision(
            start,
            b,
            r,
            row,
        )


class JointScheduler(
    BaseScheduler
):

    def __init__(
        self,
        index,
        max_batch,
        cliff_only,
    ):

        super().__init__(
            index,
            max_batch,
        )

        self.cliff_only = (
            cliff_only
        )

    def decide(
        self,
        queue,
        now,
    ):

        start = perf_counter_ns()

        best = None

        for b in self.candidates(
            queue
        ):

            slack = remaining_slack(
                queue,
                b,
                now,
            )

            configs = (
                self.index.feasible(
                    b,
                    cliff_only=
                        self.cliff_only,
                )
            )

            for row in configs:

                service = float(
                    row.latency_p95_ms
                )

                if service > slack:
                    continue

                throughput = (
                    b
                    / service
                )

                #
                # Primary:
                # max throughput.
                #
                # Tie:
                # larger batch.
                #
                # Tie:
                # higher accuracy.
                #
                score = (
                    throughput,
                    b,
                    float(
                        row.accuracy_pct
                    ),
                )

                if (
                    best is None
                    or score
                    > best[0]
                ):

                    best = (
                        score,
                        b,
                        int(row.r),
                        row,
                    )

        if best is None:

            b = min(
                self.candidates(
                    queue
                )
            )

            row = (
                self.index.dense(
                    b
                )
            )

            r = 0

        else:

            _, b, r, row = best

        return self.make_decision(
            start,
            b,
            r,
            row,
        )


def make_scheduler(
    name,
    index,
    max_batch,
):

    if name == "dense":
        return DenseScheduler(
            index,
            max_batch,
        )

    if name == "static_pomt":
        return StaticPOMTScheduler(
            index,
            max_batch,
        )

    if name == "decoupled":
        return DecoupledScheduler(
            index,
            max_batch,
        )

    if name == "joint_full":
        return JointScheduler(
            index,
            max_batch,
            cliff_only=False,
        )

    if name == "cliffserve":
        return JointScheduler(
            index,
            max_batch,
            cliff_only=True,
        )

    raise ValueError(
        name
    )


# ============================================================
# Workloads
# ============================================================

def poisson_trace(
    n,
    qps,
    deadline_ms,
    seed,
    dataset_size=10000,
):
    rng = np.random.default_rng(
        seed
    )

    inter = rng.exponential(
        1000.0 / qps,
        size=n,
    )

    arrivals = np.cumsum(
        inter
    )

    samples = rng.integers(
        0,
        dataset_size,
        size=n,
    )

    return [
        Request(
            request_id=i,
            arrival_ms=float(
                arrivals[i]
            ),
            deadline_ms=float(
                arrivals[i]
                + deadline_ms
            ),
            sample_index=int(
                samples[i]
            ),
        )

        for i in range(n)
    ]


def onoff_trace(
    n,
    qps,
    deadline_ms,
    seed,
    dataset_size=10000,
):
    rng = np.random.default_rng(
        seed
    )

    on_multiplier = 2.5
    off_multiplier = 0.25

    mean_on_ms = 100.0
    mean_off_ms = 100.0

    #
    # Scale so long-run average rate
    # remains qps.
    #
    weighted = (
        (
            mean_on_ms
            * on_multiplier
        )
        + (
            mean_off_ms
            * off_multiplier
        )
    ) / (
        mean_on_ms
        + mean_off_ms
    )

    base = (
        qps
        / weighted
    )

    on_rate = (
        base
        * on_multiplier
    )

    off_rate = (
        base
        * off_multiplier
    )

    arrivals = []

    t = 0.0
    state_on = True

    while len(arrivals) < n:

        duration = (
            rng.exponential(
                mean_on_ms
                if state_on
                else mean_off_ms
            )
        )

        state_end = (
            t
            + duration
        )

        rate = (
            on_rate
            if state_on
            else off_rate
        )

        while (
            len(arrivals)
            < n
        ):

            dt = (
                rng.exponential(
                    1000.0
                    / rate
                )
            )

            if (
                t + dt
                >= state_end
            ):
                t = state_end
                break

            t += dt

            arrivals.append(
                t
            )

        state_on = (
            not state_on
        )

    samples = rng.integers(
        0,
        dataset_size,
        size=n,
    )

    return [
        Request(
            i,
            float(arrivals[i]),
            float(
                arrivals[i]
                + deadline_ms
            ),
            int(samples[i]),
        )

        for i in range(n)
    ]


# ============================================================
# Profile-driven replay
# ============================================================

class LatencySampler:

    def __init__(
        self,
        path,
        seed,
    ):

        self.data = np.load(
            path
        )

        self.rng = (
            np.random.default_rng(
                seed
            )
        )

    def sample(
        self,
        batch_size,
        r,
    ):

        key = (
            f"b{batch_size}_r{r}"
        )

        if key not in self.data:
            raise KeyError(
                f"No latency data "
                f"for {key}"
            )

        values = self.data[
            key
        ]

        index = (
            self.rng.integers(
                0,
                len(values),
            )
        )

        return float(
            values[index]
        )


def replay(
    requests,
    scheduler,
    sampler,
    batch_window_ms,
):

    pending = []

    request_index = 0
    now = 0.0

    completed = []
    batch_records = []

    while (
        request_index
        < len(requests)
        or pending
    ):

        #
        # Nothing queued:
        # move virtual time to next arrival.
        #
        if not pending:

            now = max(
                now,
                requests[
                    request_index
                ].arrival_ms,
            )

            while (
                request_index
                < len(requests)
                and requests[
                    request_index
                ].arrival_ms
                <= now
            ):

                pending.append(
                    requests[
                        request_index
                    ]
                )

                request_index += 1

        #
        # Shared micro-batching window.
        #
        oldest = pending[0]

        dispatch_at = min(
            oldest.arrival_ms
            + batch_window_ms,

            min(
                x.deadline_ms
                for x
                in pending
            ),
        )

        while (
            len(pending)
            < scheduler.max_batch
            and request_index
            < len(requests)
        ):

            next_request = (
                requests[
                    request_index
                ]
            )

            if (
                next_request.arrival_ms
                <= dispatch_at
            ):

                pending.append(
                    next_request
                )

                request_index += 1

            else:
                break

        if now < dispatch_at:
            now = dispatch_at

        #
        # Add anything that arrived
        # while the batching timer ran.
        #
        while (
            request_index
            < len(requests)
            and requests[
                request_index
            ].arrival_ms
            <= now
        ):

            pending.append(
                requests[
                    request_index
                ]
            )

            request_index += 1

        decision = scheduler.decide(
            pending,
            now,
        )

        b = decision.batch_size

        batch = pending[:b]

        del pending[:b]

        service_ms = sampler.sample(
            b,
            decision.r,
        )

        start_ms = now

        finish_ms = (
            start_ms
            + service_ms
        )

        now = finish_ms

        batch_records.append(
            {
                "start_ms":
                    start_ms,

                "finish_ms":
                    finish_ms,

                "batch_size":
                    b,

                "r":
                    decision.r,

                "service_ms":
                    service_ms,

                "predicted_p95_ms":
                    decision
                    .predicted_service_ms,

                "estimated_accuracy_pct":
                    decision
                    .predicted_accuracy_pct,

                "scheduler_overhead_us":
                    decision
                    .scheduler_overhead_us,
            }
        )

        for req in batch:

            completed.append(
                {
                    "request_id":
                        req.request_id,

                    "arrival_ms":
                        req.arrival_ms,

                    "deadline_ms":
                        req.deadline_ms,

                    "start_ms":
                        start_ms,

                    "finish_ms":
                        finish_ms,

                    "queue_ms":
                        start_ms
                        - req.arrival_ms,

                    "service_ms":
                        service_ms,

                    "e2e_ms":
                        finish_ms
                        - req.arrival_ms,

                    "deadline_met":
                        finish_ms
                        <= req.deadline_ms,

                    "batch_size":
                        b,

                    "r":
                        decision.r,

                    "estimated_accuracy_pct":
                        decision
                        .predicted_accuracy_pct,
                }
            )

        #
        # Arrivals during inference.
        #
        while (
            request_index
            < len(requests)
            and requests[
                request_index
            ].arrival_ms
            <= now
        ):

            pending.append(
                requests[
                    request_index
                ]
            )

            request_index += 1

    return (
        pd.DataFrame(
            completed
        ),
        pd.DataFrame(
            batch_records
        ),
    )


def summarize_replay(
    completed,
    batches,
):

    beginning = float(
        completed[
            "arrival_ms"
        ].min()
    )

    end = float(
        completed[
            "finish_ms"
        ].max()
    )

    duration_s = (
        end - beginning
    ) / 1000.0

    return {
        "requests":
            len(completed),

        "slo_attainment":
            float(
                completed[
                    "deadline_met"
                ].mean()
            ),

        "throughput_qps":
            float(
                len(completed)
                / duration_s
            ),

        "e2e_p50_ms":
            float(
                completed[
                    "e2e_ms"
                ].quantile(
                    0.50
                )
            ),

        "e2e_p95_ms":
            float(
                completed[
                    "e2e_ms"
                ].quantile(
                    0.95
                )
            ),

        "e2e_p99_ms":
            float(
                completed[
                    "e2e_ms"
                ].quantile(
                    0.99
                )
            ),

        "queue_p95_ms":
            float(
                completed[
                    "queue_ms"
                ].quantile(
                    0.95
                )
            ),

        "mean_batch_size":
            float(
                batches[
                    "batch_size"
                ].mean()
            ),

        "mean_r":
            float(
                completed[
                    "r"
                ].mean()
            ),

        "estimated_accuracy_pct":
            float(
                completed[
                    "estimated_accuracy_pct"
                ].mean()
            ),

        "scheduler_overhead_mean_us":
            float(
                batches[
                    "scheduler_overhead_us"
                ].mean()
            ),

        "scheduler_overhead_p95_us":
            float(
                batches[
                    "scheduler_overhead_us"
                ].quantile(
                    0.95
                )
            ),
    }


# ============================================================
# Sweep
# ============================================================

def dense_capacity(
    surface,
    max_batch,
):
    dense = surface[
        (
            surface["r"] == 0
        )
        & (
            surface[
                "batch_size"
            ]
            <= max_batch
        )
    ].copy()

    throughput = (
        1000.0
        * dense[
            "batch_size"
        ]
        / dense[
            "latency_median_ms"
        ]
    )

    return float(
        throughput.max()
    )


def run_sweep(
    config,
    outdir,
):
    cfg = load_config(config)

    output = mkdir(
        outdir
    )

    surface = pd.read_csv(
        output
        / "surface.csv"
    )

    cliff = (
        build_cliff_frontier(
            surface,

            cfg[
                "epsilon_pp"
            ],

            cfg[
                "cliff_min_relative_gain"
            ],

            cfg[
                "cliff_min_absolute_gain_ms"
            ],
        )
    )

    cliff.to_csv(
        output
        / "cliff_frontier.csv",
        index=False,
    )

    index = SurfaceIndex(
        surface,

        epsilon_pp=
            cfg[
                "epsilon_pp"
            ],

        cliff_candidates=
            cliff,
    )

    max_batch = max(
        cfg[
            "batch_sizes"
        ]
    )

    capacity = dense_capacity(
        surface,
        max_batch,
    )

    dense_b1_p95 = float(
        surface[
            (
                surface[
                    "batch_size"
                ]
                == 1
            )
            & (
                surface[
                    "r"
                ]
                == 0
            )
        ][
            "latency_p95_ms"
        ]
        .iloc[0]
    )

    print(
        "Dense saturation capacity:",
        f"{capacity:.2f} req/s",
    )

    print(
        "Dense B=1 p95:",
        f"{dense_b1_p95:.3f} ms",
    )

    policies = [
        "dense",
        "static_pomt",
        "decoupled",
        "joint_full",
        "cliffserve",
    ]

    patterns = [
        "poisson",
        "onoff",
    ]

    summary_rows = []

    trace_dir = mkdir(
        output
        / "sweep"
    )

    for seed in cfg["seeds"]:

        for pattern in patterns:

            for load in cfg[
                "load_factors"
            ]:

                qps = (
                    capacity
                    * load
                )

                for deadline_factor in cfg[
                    "deadline_factors"
                ]:

                    deadline_ms = (
                        dense_b1_p95
                        * deadline_factor
                    )

                    if pattern == "poisson":

                        requests = poisson_trace(
                            cfg[
                                "requests_per_trace"
                            ],
                            qps,
                            deadline_ms,
                            seed,
                        )

                    else:

                        requests = onoff_trace(
                            cfg[
                                "requests_per_trace"
                            ],
                            qps,
                            deadline_ms,
                            seed,
                        )

                    for policy in policies:

                        scheduler = (
                            make_scheduler(
                                policy,
                                index,
                                max_batch,
                            )
                        )

                        sampler = (
                            LatencySampler(
                                output
                                / "latency_samples.npz",

                                seed
                                + 1000,
                            )
                        )

                        (
                            completed,
                            batches,
                        ) = replay(
                            requests,
                            scheduler,
                            sampler,
                            cfg[
                                "batch_window_ms"
                            ],
                        )

                        summary = (
                            summarize_replay(
                                completed,
                                batches,
                            )
                        )

                        summary.update(
                            {
                                "seed":
                                    seed,

                                "pattern":
                                    pattern,

                                "load_factor":
                                    load,

                                "offered_qps":
                                    qps,

                                "deadline_factor":
                                    deadline_factor,

                                "deadline_ms":
                                    deadline_ms,

                                "policy":
                                    policy,

                                "epsilon_pp":
                                    cfg[
                                        "epsilon_pp"
                                    ],
                            }
                        )

                        summary_rows.append(
                            summary
                        )

                        tag = (
                            f"{policy}_"
                            f"{pattern}_"
                            f"load{load}_"
                            f"d{deadline_factor}_"
                            f"s{seed}"
                        )

                        completed.to_csv(
                            trace_dir
                            / (
                                "requests_"
                                + tag
                                + ".csv"
                            ),

                            index=False,
                        )

                        batches.to_csv(
                            trace_dir
                            / (
                                "batches_"
                                + tag
                                + ".csv"
                            ),

                            index=False,
                        )

                        print(
                            tag,
                            "SLO=",
                            f"{summary['slo_attainment']:.3f}",
                            "throughput=",
                            f"{summary['throughput_qps']:.1f}",
                        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        trace_dir
        / "summary.csv",

        index=False,
    )


# ============================================================
# Pilot analysis
# ============================================================

def cliff_locations(
    group,
    threshold,
):
    group = group.sort_values(
        "r"
    )

    previous = None

    locations = []

    for row in group.itertuples():

        if previous is not None:

            gain = (
                float(
                    previous.latency_p95_ms
                )
                - float(
                    row.latency_p95_ms
                )
            ) / max(
                float(
                    previous.latency_p95_ms
                ),
                1e-12,
            )

            if gain >= threshold:
                locations.append(
                    int(
                        row.r
                    )
                )

        previous = row

    return locations


def run_pilot_report(
    config,
    outdir,
):
    cfg = load_config(
        config
    )

    output = Path(
        outdir
    )

    surface = pd.read_csv(
        output
        / "surface.csv"
    )

    summary_path = (
        output
        / "sweep"
        / "summary.csv"
    )

    report = {
        "batch_results": {},
    }

    best_rs = []

    cliff_sets = []

    for batch_size, group in surface.groupby(
        "batch_size"
    ):

        feasible = group[
            group[
                "accuracy_drop_pp"
            ]
            <= cfg[
                "epsilon_pp"
            ]
        ]

        if feasible.empty:

            best = group[
                group["r"] == 0
            ].iloc[0]

        else:

            qps = (
                1000.0
                * batch_size
                / feasible[
                    "latency_p95_ms"
                ]
            )

            best = feasible.loc[
                qps.idxmax()
            ]

        locations = (
            cliff_locations(
                group,
                0.05,
            )
        )

        best_rs.append(
            int(
                best["r"]
            )
        )

        cliff_sets.append(
            tuple(
                locations
            )
        )

        report[
            "batch_results"
        ][
            str(
                int(
                    batch_size
                )
            )
        ] = {
            "best_quality_feasible_r":
                int(
                    best["r"]
                ),

            "best_p95_latency_ms":
                float(
                    best[
                        "latency_p95_ms"
                    ]
                ),

            "cliff_r_values_5pct":
                locations,
        }

    any_cliff = any(
        len(x) > 0
        for x in cliff_sets
    )

    changing_cliffs = (
        len(
            set(
                cliff_sets
            )
        )
        >= 2
    )

    changing_best_r = (
        len(
            set(
                best_rs
            )
        )
        >= 2
    )

    report[
        "A_hardware_nonseparability"
    ] = {
        "any_5pct_latency_cliff":
            any_cliff,

        "cliff_locations_change_with_batch":
            changing_cliffs,

        "best_quality_feasible_r_changes_with_batch":
            changing_best_r,
    }

    #
    # Scheduler-level B test.
    #
    max_slo_gain_pp = None
    max_throughput_gain_pct = None

    if summary_path.exists():

        summary = pd.read_csv(
            summary_path
        )

        grouped = (
            summary.groupby(
                [
                    "pattern",
                    "load_factor",
                    "deadline_factor",
                    "policy",
                ],
                as_index=False,
            )
            .agg(
                slo_attainment=(
                    "slo_attainment",
                    "mean",
                ),

                throughput_qps=(
                    "throughput_qps",
                    "mean",
                ),
            )
        )

        keys = [
            "pattern",
            "load_factor",
            "deadline_factor",
        ]

        dec = (
            grouped[
                grouped[
                    "policy"
                ]
                == "decoupled"
            ]
            .drop(
                columns=[
                    "policy"
                ]
            )
            .rename(
                columns={
                    "slo_attainment":
                        "dec_slo",

                    "throughput_qps":
                        "dec_qps",
                }
            )
        )

        ours = (
            grouped[
                grouped[
                    "policy"
                ]
                == "cliffserve"
            ]
            .drop(
                columns=[
                    "policy"
                ]
            )
            .rename(
                columns={
                    "slo_attainment":
                        "ours_slo",

                    "throughput_qps":
                        "ours_qps",
                }
            )
        )

        comparison = dec.merge(
            ours,
            on=keys,
            validate="one_to_one",
        )

        comparison[
            "slo_gain_pp"
        ] = (
            100.0
            * (
                comparison[
                    "ours_slo"
                ]
                - comparison[
                    "dec_slo"
                ]
            )
        )

        comparison[
            "throughput_gain_pct"
        ] = (
            100.0
            * (
                comparison[
                    "ours_qps"
                ]
                / comparison[
                    "dec_qps"
                ]
                - 1.0
            )
        )

        max_slo_gain_pp = float(
            comparison[
                "slo_gain_pp"
            ]
            .max()
        )

        max_throughput_gain_pct = float(
            comparison[
                "throughput_gain_pct"
            ]
            .max()
        )

        comparison.to_csv(
            output
            / "joint_vs_decoupled.csv",
            index=False,
        )

    scheduler_signal = (
        max_slo_gain_pp is not None
        and (
            max_slo_gain_pp
            >= 5.0
            or max_throughput_gain_pct
            >= 10.0
        )
    )

    report[
        "B_joint_scheduler_signal"
    ] = {
        "max_slo_gain_pp":
            max_slo_gain_pp,

        "max_throughput_gain_pct":
            max_throughput_gain_pct,

        "passes_target":
            scheduler_signal,
    }

    report[
        "recommend_continue"
    ] = bool(
        any_cliff
        and (
            changing_cliffs
            or changing_best_r
        )
        and scheduler_signal
    )

    with open(
        output
        / "pilot_report.json",
        "w",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    print(
        json.dumps(
            report,
            indent=2,
        )
    )


# ============================================================
# Plots
# ============================================================

def run_plots(
    outdir,
):
    output = Path(
        outdir
    )

    figure_dir = mkdir(
        output
        / "figures"
    )

    surface = pd.read_csv(
        output
        / "surface.csv"
    )

    # ------------------------------
    # Figure 1
    # ------------------------------

    pivot = surface.pivot(
        index="batch_size",
        columns="content_tokens",
        values="latency_p95_ms",
    )

    pivot = (
        pivot
        .sort_index()
        .sort_index(
            axis=1
        )
    )

    fig, ax = plt.subplots(
        figsize=(
            8,
            4.8,
        )
    )

    image = ax.imshow(
        pivot.values,
        aspect="auto",
        origin="lower",
    )

    stride = max(
        1,
        len(
            pivot.columns
        )
        // 8,
    )

    xticks = np.arange(
        len(
            pivot.columns
        )
    )[::stride]

    ax.set_xticks(
        xticks
    )

    ax.set_xticklabels(
        [
            str(
                pivot.columns[i]
            )
            for i
            in xticks
        ],

        rotation=45,
    )

    ax.set_yticks(
        np.arange(
            len(
                pivot.index
            )
        )
    )

    ax.set_yticklabels(
        [
            str(x)
            for x
            in pivot.index
        ]
    )

    ax.set_xlabel(
        "Retained content-token budget"
    )

    ax.set_ylabel(
        "Batch size"
    )

    cbar = fig.colorbar(
        image,
        ax=ax,
    )

    cbar.set_label(
        "p95 GPU latency (ms)"
    )

    fig.tight_layout()

    fig.savefig(
        figure_dir
        / "fig1_latency_surface.pdf",

        bbox_inches="tight",
    )

    fig.savefig(
        figure_dir
        / "fig1_latency_surface.png",

        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    # ------------------------------
    # Figure 2
    # ------------------------------

    fig, ax = plt.subplots(
        figsize=(
            7.2,
            4.8,
        )
    )

    for batch_size, group in surface.groupby(
        "batch_size"
    ):

        group = group.sort_values(
            "latency_p95_ms"
        )

        ax.plot(
            group[
                "latency_p95_ms"
            ],

            group[
                "accuracy_pct"
            ],

            marker="o",

            label=(
                f"B={batch_size}"
            ),
        )

    ax.set_xlabel(
        "p95 GPU latency (ms)"
    )

    ax.set_ylabel(
        "Top-1 accuracy (%)"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        ncol=2
    )

    fig.tight_layout()

    fig.savefig(
        figure_dir
        / "fig2_batch_frontiers.pdf",

        bbox_inches="tight",
    )

    fig.savefig(
        figure_dir
        / "fig2_batch_frontiers.png",

        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    # ------------------------------
    # Scheduler plots
    # ------------------------------

    summary_path = (
        output
        / "sweep"
        / "summary.csv"
    )

    if summary_path.exists():

        summary = pd.read_csv(
            summary_path
        )

        aggregate = (
            summary.groupby(
                [
                    "pattern",
                    "deadline_factor",
                    "policy",
                    "load_factor",
                ],
                as_index=False,
            )
            .agg(
                slo_attainment=(
                    "slo_attainment",
                    "mean",
                ),

                throughput_qps=(
                    "throughput_qps",
                    "mean",
                ),

                estimated_accuracy_pct=(
                    "estimated_accuracy_pct",
                    "mean",
                ),
            )
        )

        for (
            pattern,
            deadline_factor,
        ), group in aggregate.groupby(
            [
                "pattern",
                "deadline_factor",
            ]
        ):

            fig, ax = plt.subplots(
                figsize=(
                    7.2,
                    4.8,
                )
            )

            for policy, data in group.groupby(
                "policy"
            ):

                data = data.sort_values(
                    "load_factor"
                )

                ax.plot(
                    data[
                        "load_factor"
                    ],

                    100.0
                    * data[
                        "slo_attainment"
                    ],

                    marker="o",

                    label=policy,
                )

            ax.set_xlabel(
                "Offered load / dense saturation"
            )

            ax.set_ylabel(
                "SLO attainment (%)"
            )

            ax.set_ylim(
                0,
                101,
            )

            ax.grid(
                alpha=0.25
            )

            ax.legend()

            fig.tight_layout()

            stem = (
                f"fig3_slo_"
                f"{pattern}_"
                f"d{deadline_factor}"
            )

            fig.savefig(
                figure_dir
                / f"{stem}.pdf",

                bbox_inches="tight",
            )

            fig.savefig(
                figure_dir
                / f"{stem}.png",

                dpi=220,
                bbox_inches="tight",
            )

            plt.close(
                fig
            )

    print(
        "Figures saved under",
        figure_dir,
    )


# ============================================================
# Real-image hardware validation
# ============================================================

@torch.inference_mode()
def run_live_validation(
    config,
    dataset_root,
    outdir,
    max_batches,
):
    cfg = load_config(
        config
    )

    output = Path(
        outdir
    )

    frontier = pd.read_csv(
        output
        / "cliff_frontier.csv"
    )

    configs = (
        frontier[
            [
                "batch_size",
                "r",
            ]
        ]
        .drop_duplicates()
    )

    dense, pomt, transform = (
        make_models(
            cfg[
                "model"
            ],

            cfg[
                "device"
            ],

            cfg[
                "prune_layer"
            ],
        )
    )

    dataset = (
        ImageNetV2NumericFolders(
            dataset_root,
            transform,
        )
    )

    rows = []

    for config_row in configs.itertuples():

        b = int(
            config_row.batch_size
        )

        r = int(
            config_row.r
        )

        print(
            "Live validation:",
            "B=",
            b,
            "R=",
            r,
        )

        if r == 0:
            model = dense

        else:
            set_pomt_r(
                pomt,
                r,
                cfg[
                    "prune_layer"
                ],
            )

            model = pomt

        loader = DataLoader(
            dataset,
            batch_size=b,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

        latencies = []

        correct = 0
        total = 0

        for (
            batch_index,
            (
                images,
                targets,
                _,
            ),
        ) in enumerate(
            loader
        ):

            if batch_index >= max_batches:
                break

            images = images.to(
                cfg[
                    "device"
                ],
                non_blocking=True,
            )

            targets = targets.to(
                cfg[
                    "device"
                ],
                non_blocking=True,
            )

            if batch_index == 0:

                model(
                    images
                )

                torch.cuda.synchronize()

            begin = torch.cuda.Event(
                enable_timing=True
            )

            end = torch.cuda.Event(
                enable_timing=True
            )

            begin.record()

            logits = model(
                images
            )

            end.record()

            torch.cuda.synchronize()

            latencies.append(
                float(
                    begin.elapsed_time(
                        end
                    )
                )
            )

            predictions = (
                logits.argmax(
                    dim=1
                )
            )

            correct += int(
                (
                    predictions
                    == targets
                )
                .sum()
                .item()
            )

            total += int(
                targets.numel()
            )

        values = np.asarray(
            latencies
        )

        rows.append(
            {
                "batch_size":
                    b,

                "r":
                    r,

                "real_input_median_ms":
                    float(
                        np.median(
                            values
                        )
                    ),

                "real_input_p95_ms":
                    float(
                        np.percentile(
                            values,
                            95,
                        )
                    ),

                "real_input_p99_ms":
                    float(
                        np.percentile(
                            values,
                            99,
                        )
                    ),

                "real_input_accuracy_pct":
                    (
                        100.0
                        * correct
                        / total
                    ),

                "images":
                    total,
            }
        )

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        output
        / "live_validation.csv",

        index=False,
    )

    print(
        result.to_string(
            index=False
        )
    )


# ============================================================
# Full config generator
# ============================================================

def make_full_config(
    pilot_config,
    output_path,
):
    cfg = load_config(
        pilot_config
    )

    cfg[
        "r_values"
    ] = list(
        range(
            0,
            153,
            8,
        )
    )

    cfg[
        "latency_warmup"
    ] = 50

    cfg[
        "latency_repeats"
    ] = 300

    cfg[
        "accuracy_samples"
    ] = 10000

    cfg[
        "seeds"
    ] = [
        84,
        85,
        86,
        87,
        88,
    ]

    cfg[
        "requests_per_trace"
    ] = 10000

    with open(
        output_path,
        "w",
    ) as f:

        yaml.safe_dump(
            cfg,
            f,
            sort_keys=False,
        )

    print(
        "Saved",
        output_path,
    )



# ============================================================
# Extended latency-only characterization
# ============================================================

def run_latency_only(config, outdir):
    """
    Profile GPU latency only, without loading ImageNetV2 or evaluating accuracy.

    Intended for quickly expanding the batch-size/token-pruning hardware surface.
    Existing latency samples can optionally be merged later.
    """

    cfg = load_config(config)

    seed_all(84)

    output = mkdir(outdir)

    provenance = hardware_provenance(
        cfg["device"]
    )

    provenance["config"] = cfg
    provenance["experiment_type"] = "latency_only"

    with open(
        output / "provenance.json",
        "w",
    ) as f:
        json.dump(
            provenance,
            f,
            indent=2,
            default=str,
        )

    dense, pomt, _ = make_models(
        cfg["model"],
        cfg["device"],
        cfg["prune_layer"],
    )

    latency = profile_latency(
        cfg,
        dense,
        pomt,
        output,
    )

    print()
    print("=" * 80)
    print("EXTENDED LATENCY CHARACTERIZATION COMPLETE")
    print("=" * 80)

    print(
        latency[
            [
                "batch_size",
                "r",
                "content_tokens",
                "latency_median_ms",
                "latency_p95_ms",
                "throughput_median_qps",
                "throughput_p95_qps",
            ]
        ].to_string(index=False)
    )


def merge_latency_surfaces(
    pilot_dir,
    extended_dir,
    output_path,
):
    """
    Merge the original pilot latency characterization and the extended
    batch-size latency characterization into one CSV.
    """

    pilot_path = (
        Path(pilot_dir)
        / "latency.csv"
    )

    extended_path = (
        Path(extended_dir)
        / "latency.csv"
    )

    pilot = pd.read_csv(
        pilot_path
    )

    extended = pd.read_csv(
        extended_path
    )

    merged = pd.concat(
        [
            pilot,
            extended,
        ],
        ignore_index=True,
    )

    merged = (
        merged
        .drop_duplicates(
            subset=[
                "model",
                "batch_size",
                "r",
            ],
            keep="last",
        )
        .sort_values(
            [
                "batch_size",
                "r",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Merged latency surface saved to {output_path}"
    )

    print(
        f"Rows: {len(merged)}"
    )


def analyze_extended_latency(
    latency_path,
    output_dir,
    cliff_threshold=0.05,
):
    """
    Analyze hardware-only latency behavior before running any additional
    ImageNetV2 accuracy evaluation.

    Produces:
      - best throughput configuration per batch
      - relative speedup vs dense
      - detected >= threshold latency drops
      - batch x pruning latency heatmap
      - throughput curves
    """

    output_dir = mkdir(
        output_dir
    )

    latency = pd.read_csv(
        latency_path
    )

    rows = []
    cliffs = []

    for batch_size, group in latency.groupby(
        "batch_size"
    ):

        group = group.sort_values(
            "r"
        ).copy()

        dense_row = group[
            group["r"] == 0
        ]

        if dense_row.empty:
            raise RuntimeError(
                f"No dense R=0 measurement for B={batch_size}"
            )

        dense_p95 = float(
            dense_row[
                "latency_p95_ms"
            ].iloc[0]
        )

        dense_qps = (
            1000.0
            * batch_size
            / dense_p95
        )

        group[
            "speedup_vs_dense_p95"
        ] = (
            dense_p95
            / group[
                "latency_p95_ms"
            ]
        )

        group[
            "throughput_vs_dense_pct"
        ] = (
            (
                group[
                    "throughput_p95_qps"
                ]
                / dense_qps
            )
            - 1.0
        ) * 100.0

        best_index = (
            group[
                "throughput_p95_qps"
            ].idxmax()
        )

        best = group.loc[
            best_index
        ]

        rows.append(
            {
                "batch_size":
                    int(batch_size),

                "dense_p95_ms":
                    dense_p95,

                "dense_p95_qps":
                    dense_qps,

                "best_r":
                    int(
                        best["r"]
                    ),

                "best_content_tokens":
                    int(
                        best[
                            "content_tokens"
                        ]
                    ),

                "best_p95_ms":
                    float(
                        best[
                            "latency_p95_ms"
                        ]
                    ),

                "best_p95_qps":
                    float(
                        best[
                            "throughput_p95_qps"
                        ]
                    ),

                "best_speedup_vs_dense":
                    float(
                        best[
                            "speedup_vs_dense_p95"
                        ]
                    ),

                "best_throughput_gain_pct":
                    float(
                        best[
                            "throughput_vs_dense_pct"
                        ]
                    ),
            }
        )

        previous = None

        for row in group.itertuples():

            if previous is not None:

                previous_latency = float(
                    previous.latency_p95_ms
                )

                current_latency = float(
                    row.latency_p95_ms
                )

                gain = (
                    previous_latency
                    - current_latency
                ) / max(
                    previous_latency,
                    1e-12,
                )

                if gain >= cliff_threshold:

                    cliffs.append(
                        {
                            "batch_size":
                                int(
                                    batch_size
                                ),

                            "from_r":
                                int(
                                    previous.r
                                ),

                            "to_r":
                                int(
                                    row.r
                                ),

                            "from_content_tokens":
                                int(
                                    previous.content_tokens
                                ),

                            "to_content_tokens":
                                int(
                                    row.content_tokens
                                ),

                            "previous_p95_ms":
                                previous_latency,

                            "current_p95_ms":
                                current_latency,

                            "relative_latency_drop_pct":
                                100.0
                                * gain,
                        }
                    )

            previous = row

    summary = pd.DataFrame(
        rows
    )

    cliff_df = pd.DataFrame(
        cliffs
    )

    summary.to_csv(
        output_dir
        / "best_latency_configuration_by_batch.csv",
        index=False,
    )

    cliff_df.to_csv(
        output_dir
        / "detected_latency_cliffs.csv",
        index=False,
    )

    print()
    print("=" * 80)
    print("BEST HARDWARE CONFIGURATION PER BATCH")
    print("=" * 80)

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print(
        f"DETECTED >= {100*cliff_threshold:.1f}% P95 LATENCY CLIFFS"
    )
    print("=" * 80)

    if cliff_df.empty:
        print(
            "No cliffs above threshold."
        )
    else:
        print(
            cliff_df.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Heatmap
    # --------------------------------------------------------

    pivot = latency.pivot(
        index="batch_size",
        columns="r",
        values="latency_p95_ms",
    )

    pivot = (
        pivot
        .sort_index()
        .sort_index(
            axis=1
        )
    )

    fig, ax = plt.subplots(
        figsize=(
            9.5,
            5.4,
        )
    )

    image = ax.imshow(
        pivot.values,
        aspect="auto",
        origin="lower",
    )

    ax.set_xticks(
        np.arange(
            len(
                pivot.columns
            )
        )
    )

    ax.set_xticklabels(
        [
            str(x)
            for x
            in pivot.columns
        ],
        rotation=45,
    )

    ax.set_yticks(
        np.arange(
            len(
                pivot.index
            )
        )
    )

    ax.set_yticklabels(
        [
            str(x)
            for x
            in pivot.index
        ]
    )

    ax.set_xlabel(
        "Pruned tokens R"
    )

    ax.set_ylabel(
        "Batch size"
    )

    cbar = fig.colorbar(
        image,
        ax=ax,
    )

    cbar.set_label(
        "p95 GPU latency (ms)"
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "extended_latency_heatmap.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        output_dir
        / "extended_latency_heatmap.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    # --------------------------------------------------------
    # Throughput curves
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(
            8.0,
            5.0,
        )
    )

    for batch_size, group in latency.groupby(
        "batch_size"
    ):

        group = group.sort_values(
            "r"
        )

        ax.plot(
            group["r"],
            group[
                "throughput_p95_qps"
            ],
            marker="o",
            label=f"B={batch_size}",
        )

    ax.set_xlabel(
        "Pruned tokens R"
    )

    ax.set_ylabel(
        "p95-based throughput (images/s)"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        ncol=3,
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "extended_throughput_curves.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        output_dir
        / "extended_throughput_curves.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )




# ============================================================
# Accuracy-only characterization
# ============================================================

def run_accuracy_only(
    config,
    dataset_root,
    outdir,
):
    """
    Evaluate ImageNetV2 accuracy over the requested (B, R) grid
    without repeating GPU latency profiling.
    """

    cfg = load_config(config)

    seed_all(84)

    output = mkdir(outdir)

    provenance = hardware_provenance(
        cfg["device"]
    )

    provenance["config"] = cfg
    provenance["experiment_type"] = "accuracy_only"

    with open(
        output / "provenance.json",
        "w",
    ) as f:
        json.dump(
            provenance,
            f,
            indent=2,
            default=str,
        )

    dense, pomt, transform = make_models(
        cfg["model"],
        cfg["device"],
        cfg["prune_layer"],
    )

    dataset = ImageNetV2NumericFolders(
        dataset_root,
        transform=transform,
    )

    print()
    print("=" * 80)
    print("10K IMAGENETV2 ACCURACY CHARACTERIZATION")
    print("=" * 80)

    print(
        "Dataset size:",
        len(dataset),
    )

    print(
        "Samples requested:",
        cfg["accuracy_samples"],
    )

    print(
        "Batch sizes:",
        cfg["batch_sizes"],
    )

    print(
        "R values:",
        cfg["r_values"],
    )

    accuracy = evaluate_accuracy(
        cfg,
        dense,
        pomt,
        dataset,
        output,
    )

    print()
    print("=" * 80)
    print("ACCURACY CHARACTERIZATION COMPLETE")
    print("=" * 80)

    print(
        accuracy[
            [
                "batch_size",
                "r",
                "accuracy_pct",
                "correct",
                "num_samples",
            ]
        ].to_string(
            index=False
        )
    )


def build_final_quality_surface(
    latency_path,
    accuracy_path,
    output_dir,
):
    """
    Merge the previously measured GPU latency surface with the
    full ImageNetV2 accuracy characterization.
    """

    output_dir = mkdir(
        output_dir
    )

    latency = pd.read_csv(
        latency_path
    )

    accuracy = pd.read_csv(
        accuracy_path
    )

    keys = [
        "model",
        "batch_size",
        "r",
    ]

    latency_keys = set(
        tuple(x)
        for x
        in latency[keys].itertuples(
            index=False,
            name=None,
        )
    )

    accuracy_keys = set(
        tuple(x)
        for x
        in accuracy[keys].itertuples(
            index=False,
            name=None,
        )
    )

    missing_latency = (
        accuracy_keys
        - latency_keys
    )

    if missing_latency:
        raise RuntimeError(
            "Accuracy configurations are missing latency measurements:\n"
            + "\n".join(
                str(x)
                for x
                in sorted(
                    missing_latency
                )
            )
        )

    #
    # Keep only latency configurations that were evaluated
    # in the 10k accuracy experiment.
    #
    latency = latency.merge(
        accuracy[keys],
        on=keys,
        how="inner",
    )

    surface = build_surface(
        latency,
        accuracy,
    )

    surface.to_csv(
        output_dir
        / "quality_latency_surface.csv",
        index=False,
    )

    print()
    print("=" * 80)
    print("FINAL QUALITY-LATENCY SURFACE")
    print("=" * 80)

    print(
        surface[
            [
                "batch_size",
                "r",
                "content_tokens",
                "latency_p95_ms",
                "accuracy_pct",
                "accuracy_drop_pp",
                "speedup_vs_dense_p95",
                "pareto",
            ]
        ].to_string(
            index=False
        )
    )


def analyze_quality_budgets(
    surface_path,
    output_dir,
    epsilons,
):
    """
    Extract the best p95-latency / throughput operating point
    per batch for each requested accuracy-loss budget.
    """

    output_dir = mkdir(
        output_dir
    )

    surface = pd.read_csv(
        surface_path
    )

    all_rows = []

    for epsilon_pp in epsilons:

        rows = []

        for batch_size, group in surface.groupby(
            "batch_size"
        ):

            feasible = group[
                group[
                    "accuracy_drop_pp"
                ]
                <= epsilon_pp
            ].copy()

            if feasible.empty:
                continue

            #
            # Min p95 latency = max throughput for fixed B.
            #
            best = feasible.loc[
                feasible[
                    "latency_p95_ms"
                ].idxmin()
            ]

            dense = group[
                group["r"] == 0
            ].iloc[0]

            dense_latency = float(
                dense[
                    "latency_p95_ms"
                ]
            )

            best_latency = float(
                best[
                    "latency_p95_ms"
                ]
            )

            speedup = (
                dense_latency
                / best_latency
            )

            throughput_gain_pct = (
                speedup - 1.0
            ) * 100.0

            row = {
                "epsilon_pp":
                    float(
                        epsilon_pp
                    ),

                "batch_size":
                    int(
                        batch_size
                    ),

                "best_r":
                    int(
                        best["r"]
                    ),

                "best_content_tokens":
                    int(
                        best[
                            "content_tokens"
                        ]
                    ),

                "accuracy_pct":
                    float(
                        best[
                            "accuracy_pct"
                        ]
                    ),

                "dense_accuracy_pct":
                    float(
                        best[
                            "dense_accuracy_pct"
                        ]
                    ),

                "accuracy_drop_pp":
                    float(
                        best[
                            "accuracy_drop_pp"
                        ]
                    ),

                "dense_p95_ms":
                    dense_latency,

                "best_p95_ms":
                    best_latency,

                "speedup_vs_dense":
                    speedup,

                "throughput_gain_pct":
                    throughput_gain_pct,
            }

            rows.append(
                row
            )

            all_rows.append(
                row
            )

        result = pd.DataFrame(
            rows
        )

        result.to_csv(
            output_dir
            / (
                f"best_configs_epsilon_"
                f"{epsilon_pp:g}pp.csv"
            ),
            index=False,
        )

        print()
        print("=" * 80)
        print(
            f"BEST CONFIGURATIONS: epsilon={epsilon_pp:g} pp"
        )
        print("=" * 80)

        if not result.empty:
            print(
                result.to_string(
                    index=False,
                    float_format=lambda x:
                        f"{x:.4f}",
                )
            )

    combined = pd.DataFrame(
        all_rows
    )

    combined.to_csv(
        output_dir
        / "best_configs_all_epsilons.csv",
        index=False,
    )


def plot_quality_constrained_gain(
    summary_path,
    output_dir,
):
    """
    Paper-style plot:
    throughput gain versus batch size for each accuracy budget.
    """

    output_dir = mkdir(
        output_dir
    )

    data = pd.read_csv(
        summary_path
    )

    fig, ax = plt.subplots(
        figsize=(
            7.5,
            4.8,
        )
    )

    for epsilon_pp, group in data.groupby(
        "epsilon_pp"
    ):

        group = group.sort_values(
            "batch_size"
        )

        ax.plot(
            group[
                "batch_size"
            ],

            group[
                "throughput_gain_pct"
            ],

            marker="o",

            label=(
                rf"$\epsilon={epsilon_pp:g}$ pp"
            ),
        )

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.set_xlabel(
        "Batch size"
    )

    ax.set_ylabel(
        "Throughput gain vs. dense (%)"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "quality_constrained_throughput_gain.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        output_dir
        / "quality_constrained_throughput_gain.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        "Saved quality-constrained gain figure."
    )



# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "CliffServe-ViT "
            "DATE 2027 research prototype"
        )
    )

    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # extended latency-only profiling

    p = sub.add_parser(
        "latency-only"
    )

    p.add_argument(
        "--config",
        required=True,
    )

    p.add_argument(
        "--outdir",
        required=True,
    )

    # merge latency profiles

    p = sub.add_parser(
        "merge-latency"
    )

    p.add_argument(
        "--pilot-dir",
        required=True,
    )

    p.add_argument(
        "--extended-dir",
        required=True,
    )

    p.add_argument(
        "--output",
        required=True,
    )

    # analyze latency surface

    p = sub.add_parser(
        "analyze-latency"
    )

    p.add_argument(
        "--latency",
        required=True,
    )

    p.add_argument(
        "--outdir",
        required=True,
    )

    p.add_argument(
        "--cliff-threshold",
        type=float,
        default=0.05,
    )

    # 10k accuracy-only characterization

    p = sub.add_parser(
        "accuracy-only"
    )

    p.add_argument(
        "--config",
        required=True,
    )

    p.add_argument(
        "--dataset-root",
        required=True,
    )

    p.add_argument(
        "--outdir",
        required=True,
    )

    # merge measured latency + accuracy

    p = sub.add_parser(
        "build-quality-surface"
    )

    p.add_argument(
        "--latency",
        required=True,
    )

    p.add_argument(
        "--accuracy",
        required=True,
    )

    p.add_argument(
        "--outdir",
        required=True,
    )

    # quality budget analysis

    p = sub.add_parser(
        "quality-budgets"
    )

    p.add_argument(
        "--surface",
        required=True,
    )

    p.add_argument(
        "--outdir",
        required=True,
    )

    p.add_argument(
        "--epsilons",
        nargs="+",
        type=float,
        default=[
            0.5,
            1.0,
            2.0,
        ],
    )

    # quality-constrained plot

    p = sub.add_parser(
        "plot-quality-gain"
    )

    p.add_argument(
        "--summary",
        required=True,
    )

    p.add_argument(
        "--outdir",
        required=True,
    )

    # sanity

    p = sub.add_parser(
        "sanity"
    )

    p.add_argument(
        "--config",
        default="pilot.yaml",
    )

    # profile

    p = sub.add_parser(
        "profile"
    )

    p.add_argument(
        "--config",
        default="pilot.yaml",
    )

    p.add_argument(
        "--dataset-root",
        required=True,
    )

    p.add_argument(
        "--outdir",
        default="results/pilot",
    )

    # sweep

    p = sub.add_parser(
        "sweep"
    )

    p.add_argument(
        "--config",
        default="pilot.yaml",
    )

    p.add_argument(
        "--outdir",
        default="results/pilot",
    )

    # report

    p = sub.add_parser(
        "pilot-report"
    )

    p.add_argument(
        "--config",
        default="pilot.yaml",
    )

    p.add_argument(
        "--outdir",
        default="results/pilot",
    )

    # plots

    p = sub.add_parser(
        "plot"
    )

    p.add_argument(
        "--outdir",
        default="results/pilot",
    )

    # live validate

    p = sub.add_parser(
        "live-validate"
    )

    p.add_argument(
        "--config",
        default="pilot.yaml",
    )

    p.add_argument(
        "--dataset-root",
        required=True,
    )

    p.add_argument(
        "--outdir",
        default="results/pilot",
    )

    p.add_argument(
        "--max-batches",
        type=int,
        default=50,
    )

    # make full config

    p = sub.add_parser(
        "make-full-config"
    )

    p.add_argument(
        "--pilot-config",
        default="pilot.yaml",
    )

    p.add_argument(
        "--output",
        default="full.yaml",
    )

    args = parser.parse_args()

    if args.command == "accuracy-only":

        run_accuracy_only(
            args.config,
            args.dataset_root,
            args.outdir,
        )

    elif args.command == "build-quality-surface":

        build_final_quality_surface(
            args.latency,
            args.accuracy,
            args.outdir,
        )

    elif args.command == "quality-budgets":

        analyze_quality_budgets(
            args.surface,
            args.outdir,
            args.epsilons,
        )

    elif args.command == "plot-quality-gain":

        plot_quality_constrained_gain(
            args.summary,
            args.outdir,
        )

    elif args.command == "latency-only":

        run_latency_only(
            args.config,
            args.outdir,
        )

    elif args.command == "merge-latency":

        merge_latency_surfaces(
            args.pilot_dir,
            args.extended_dir,
            args.output,
        )

    elif args.command == "analyze-latency":

        analyze_extended_latency(
            args.latency,
            args.outdir,
            cliff_threshold=args.cliff_threshold,
        )

    elif args.command == "sanity":

        run_sanity(
            args.config
        )

    elif args.command == "profile":

        run_profile(
            args.config,
            args.dataset_root,
            args.outdir,
        )

    elif args.command == "sweep":

        run_sweep(
            args.config,
            args.outdir,
        )

    elif args.command == "pilot-report":

        run_pilot_report(
            args.config,
            args.outdir,
        )

    elif args.command == "plot":

        run_plots(
            args.outdir
        )

    elif args.command == "live-validate":

        run_live_validation(
            args.config,
            args.dataset_root,
            args.outdir,
            args.max_batches,
        )

    elif args.command == "make-full-config":

        make_full_config(
            args.pilot_config,
            args.output,
        )


if __name__ == "__main__":
    main()
