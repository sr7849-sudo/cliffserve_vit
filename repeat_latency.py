#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cliffserve import (
    make_models,
    measure_cuda_latency,
    set_pomt_r,
)


MODEL = "deit_small_patch16_224"
DEVICE = "cuda:0"
PRUNE_LAYER = 3

BATCHES = [
    8,
    16,
    32,
    64,
]

R_VALUES = [
    0,
    64,
    96,
]

ROUNDS = 5

WARMUP = 50
REPEATS = 300


outdir = Path(
    "results/latency_variability"
)

outdir.mkdir(
    parents=True,
    exist_ok=True,
)


if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA unavailable"
    )


dense, pomt, _ = make_models(
    MODEL,
    DEVICE,
    PRUNE_LAYER,
)


rows = []


for round_id in range(ROUNDS):

    print()
    print(
        "=" * 70
    )

    print(
        "ROUND",
        round_id + 1,
        "/",
        ROUNDS,
    )

    print(
        "=" * 70
    )

    for b in BATCHES:

        for r in R_VALUES:

            if r == 0:
                model = dense

            else:
                set_pomt_r(
                    pomt,
                    r,
                    PRUNE_LAYER,
                )

                model = pomt

            samples = measure_cuda_latency(
                model=model,
                batch_size=b,
                device=DEVICE,
                warmup=WARMUP,
                repeats=REPEATS,
            )

            median = float(
                np.median(samples)
            )

            p95 = float(
                np.percentile(
                    samples,
                    95,
                )
            )

            p99 = float(
                np.percentile(
                    samples,
                    99,
                )
            )

            rows.append(
                {
                    "round":
                        round_id,

                    "batch_size":
                        b,

                    "r":
                        r,

                    "median_ms":
                        median,

                    "p95_ms":
                        p95,

                    "p99_ms":
                        p99,

                    "mean_ms":
                        float(
                            np.mean(samples)
                        ),

                    "std_ms":
                        float(
                            np.std(
                                samples,
                                ddof=1,
                            )
                        ),
                }
            )

            print(
                f"B={b:>2} "
                f"R={r:>3} "
                f"median={median:.3f} "
                f"p95={p95:.3f}"
            )


raw = pd.DataFrame(
    rows
)

raw.to_csv(
    outdir
    / "repeated_latency_raw.csv",
    index=False,
)


summary = (
    raw.groupby(
        [
            "batch_size",
            "r",
        ],
        as_index=False,
    )
    .agg(
        median_mean_ms=(
            "median_ms",
            "mean",
        ),

        median_std_ms=(
            "median_ms",
            "std",
        ),

        p95_mean_ms=(
            "p95_ms",
            "mean",
        ),

        p95_std_ms=(
            "p95_ms",
            "std",
        ),

        p99_mean_ms=(
            "p99_ms",
            "mean",
        ),

        p99_std_ms=(
            "p99_ms",
            "std",
        ),
    )
)


summary[
    "p95_cv_pct"
] = (
    100.0
    * summary[
        "p95_std_ms"
    ]
    / summary[
        "p95_mean_ms"
    ]
)


summary[
    "p95_ci95_halfwidth_ms"
] = (
    1.96
    * summary[
        "p95_std_ms"
    ]
    / np.sqrt(
        ROUNDS
    )
)


summary.to_csv(
    outdir
    / "repeated_latency_summary.csv",
    index=False,
)


print()
print("=" * 80)
print("REPEATED LATENCY SUMMARY")
print("=" * 80)

print(
    summary.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)
