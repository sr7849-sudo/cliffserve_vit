#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
import yaml

from cliffserve import (
    Decision,
    LatencySampler,
    Request,
    poisson_trace,
    onoff_trace,
    replay,
    summarize_replay,
    build_cliff_frontier,
)


# ============================================================
# FAST PRECOMPUTED OPERATING TABLE
# ============================================================

@dataclass(frozen=True)
class Point:
    batch_size: int
    r: int
    latency_p95_ms: float
    accuracy_pct: float
    accuracy_drop_pp: float


class FastSurface:
    """
    Precomputed serving LUT.

    No DataFrame scans happen in the scheduler's hot path.
    """

    def __init__(
        self,
        surface: pd.DataFrame,
        epsilon_pp: float,
        cliff_min_relative_gain: float,
        cliff_min_absolute_gain_ms: float,
    ):
        self.epsilon_pp = float(epsilon_pp)

        self.batch_sizes = sorted(
            int(x)
            for x in surface["batch_size"].unique()
        )

        self.points = {}
        self.dense = {}
        self.feasible = {}

        # ----------------------------------------------------
        # Full quality-feasible points
        # ----------------------------------------------------

        for b in self.batch_sizes:

            g = surface[
                surface["batch_size"] == b
            ].copy()

            pts = []

            for row in g.itertuples():

                point = Point(
                    batch_size=int(row.batch_size),
                    r=int(row.r),
                    latency_p95_ms=float(row.latency_p95_ms),
                    accuracy_pct=float(row.accuracy_pct),
                    accuracy_drop_pp=float(row.accuracy_drop_pp),
                )

                self.points[(point.batch_size, point.r)] = point

                if point.r == 0:
                    self.dense[b] = point

                if point.accuracy_drop_pp <= self.epsilon_pp + 1e-12:
                    pts.append(point)

            self.feasible[b] = tuple(pts)

        # ----------------------------------------------------
        # Hardware cliff frontier
        # ----------------------------------------------------

        cliff = build_cliff_frontier(
            surface=surface,
            epsilon_pp=self.epsilon_pp,
            min_relative_gain=cliff_min_relative_gain,
            min_absolute_gain_ms=cliff_min_absolute_gain_ms,
        )

        self.cliff = {}

        for b in self.batch_sizes:

            pts = []

            g = cliff[
                cliff["batch_size"] == b
            ]

            for row in g.itertuples():

                key = (
                    int(row.batch_size),
                    int(row.r),
                )

                pts.append(
                    self.points[key]
                )

            # Dense must always remain available.
            if not any(p.r == 0 for p in pts):
                pts.append(self.dense[b])

            self.cliff[b] = tuple(pts)


# ============================================================
# HELPERS
# ============================================================

def slack_for_prefix(queue, b, now):
    return min(
        request.deadline_ms - now
        for request in queue[:b]
    )


class FastSchedulerBase:

    def __init__(
        self,
        surface: FastSurface,
        max_batch: int,
    ):
        self.surface = surface
        self.max_batch = int(max_batch)

    def batch_candidates(self, queue):
        n = min(
            len(queue),
            self.max_batch,
        )

        return [
            b
            for b in self.surface.batch_sizes
            if b <= n
        ]

    @staticmethod
    def result(
        start_ns,
        b,
        point,
    ):
        return Decision(
            batch_size=int(b),
            r=int(point.r),
            predicted_service_ms=float(
                point.latency_p95_ms
            ),
            predicted_accuracy_pct=float(
                point.accuracy_pct
            ),
            scheduler_overhead_us=(
                perf_counter_ns()
                - start_ns
            ) / 1000.0,
        )


# ============================================================
# DENSE
# ============================================================

class DenseFast(FastSchedulerBase):

    def decide(self, queue, now):

        start = perf_counter_ns()

        batches = self.batch_candidates(queue)

        valid = []

        for b in batches:

            p = self.surface.dense[b]

            if p.latency_p95_ms <= slack_for_prefix(
                queue,
                b,
                now,
            ):
                valid.append((b, p))

        if valid:
            b, p = max(
                valid,
                key=lambda x: x[0],
            )
        else:
            b = min(batches)
            p = self.surface.dense[b]

        return self.result(
            start,
            b,
            p,
        )


# ============================================================
# STATIC PRUNING
# ============================================================

def choose_static_r(surface: FastSurface):

    feasible_sets = []

    for b in surface.batch_sizes:
        feasible_sets.append(
            {
                p.r
                for p in surface.feasible[b]
            }
        )

    common = set.intersection(
        *feasible_sets
    )

    if not common:
        return 0

    best_r = 0
    best_score = -np.inf

    for r in sorted(common):

        qps = []

        for b in surface.batch_sizes:

            p = surface.points[(b, r)]

            qps.append(
                1000.0
                * b
                / p.latency_p95_ms
            )

        score = float(
            np.mean(qps)
        )

        if score > best_score:
            best_score = score
            best_r = int(r)

    return best_r


class StaticFast(FastSchedulerBase):

    def __init__(
        self,
        surface,
        max_batch,
    ):
        super().__init__(
            surface,
            max_batch,
        )

        self.static_r = choose_static_r(
            surface
        )

    def decide(self, queue, now):

        start = perf_counter_ns()

        batches = self.batch_candidates(queue)

        valid = []

        for b in batches:

            p = self.surface.points[
                (b, self.static_r)
            ]

            if p.latency_p95_ms <= slack_for_prefix(
                queue,
                b,
                now,
            ):
                valid.append((b, p))

        if valid:
            b, p = max(
                valid,
                key=lambda x: x[0],
            )
        else:
            b = min(batches)
            p = self.surface.dense[b]

        return self.result(
            start,
            b,
            p,
        )


# ============================================================
# DECOUPLED
# ============================================================

class DecoupledFast(FastSchedulerBase):

    def decide(self, queue, now):

        start = perf_counter_ns()

        batches = self.batch_candidates(queue)

        # Stage 1:
        # choose batch size using dense execution only.
        dense_valid = []

        for b in batches:

            p = self.surface.dense[b]

            if p.latency_p95_ms <= slack_for_prefix(
                queue,
                b,
                now,
            ):
                dense_valid.append(b)

        if dense_valid:
            b = max(dense_valid)
        else:
            b = min(batches)

        slack = slack_for_prefix(
            queue,
            b,
            now,
        )

        # Stage 2:
        # after B is fixed, select fastest quality-feasible R.
        valid = [
            p
            for p in self.surface.feasible[b]
            if p.latency_p95_ms <= slack
        ]

        if valid:
            p = min(
                valid,
                key=lambda p:
                    p.latency_p95_ms,
            )
        else:
            p = self.surface.dense[b]

        return self.result(
            start,
            b,
            p,
        )


# ============================================================
# JOINT
# ============================================================

class JointFast(FastSchedulerBase):

    def __init__(
        self,
        surface,
        max_batch,
        cliff_only,
    ):
        super().__init__(
            surface,
            max_batch,
        )

        self.cliff_only = bool(
            cliff_only
        )

    def decide(self, queue, now):

        start = perf_counter_ns()

        best = None

        for b in self.batch_candidates(queue):

            slack = slack_for_prefix(
                queue,
                b,
                now,
            )

            candidates = (
                self.surface.cliff[b]
                if self.cliff_only
                else self.surface.feasible[b]
            )

            for p in candidates:

                if p.latency_p95_ms > slack:
                    continue

                qps = (
                    1000.0
                    * b
                    / p.latency_p95_ms
                )

                score = (
                    qps,
                    b,
                    p.accuracy_pct,
                )

                if (
                    best is None
                    or score > best[0]
                ):
                    best = (
                        score,
                        b,
                        p,
                    )

        if best is None:

            b = min(
                self.batch_candidates(queue)
            )

            p = self.surface.dense[b]

        else:

            _, b, p = best

        return self.result(
            start,
            b,
            p,
        )


# ============================================================
# CONSTRUCTION
# ============================================================

def build_scheduler(
    policy,
    surface,
    max_batch,
):

    if policy == "dense":
        return DenseFast(
            surface,
            max_batch,
        )

    if policy == "static_pomt":
        return StaticFast(
            surface,
            max_batch,
        )

    if policy == "decoupled":
        return DecoupledFast(
            surface,
            max_batch,
        )

    if policy == "joint_full":
        return JointFast(
            surface,
            max_batch,
            cliff_only=False,
        )

    if policy == "cliffserve":
        return JointFast(
            surface,
            max_batch,
            cliff_only=True,
        )

    raise ValueError(policy)


# ============================================================
# CHECKPOINT / RESUME
# ============================================================

CHECKPOINT_COLUMNS = [
    "epsilon_pp",
    "seed",
    "pattern",
    "load_factor",
    "deadline_factor",
    "policy",
]


def scenario_key(row):
    return (
        float(row["epsilon_pp"]),
        int(row["seed"]),
        str(row["pattern"]),
        float(row["load_factor"]),
        float(row["deadline_factor"]),
        str(row["policy"]),
    )


def load_checkpoint(path):
    path = Path(path)

    if not path.exists():
        return pd.DataFrame(), set()

    df = pd.read_csv(path)

    completed = {
        scenario_key(row)
        for _, row in df.iterrows()
    }

    print(
        f"Resume checkpoint: {len(df)} completed scenarios"
    )

    return df, completed


def append_checkpoint(path, row):
    path = Path(path)

    df = pd.DataFrame([row])

    df.to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
    )



# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--surface",
        required=True,
    )

    parser.add_argument(
        "--latency-samples",
        required=True,
    )

    parser.add_argument(
        "--outdir",
        required=True,
    )

    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    surface_df = pd.read_csv(
        args.surface
    )

    outdir = Path(args.outdir)
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    max_batch = int(
        max(
            surface_df["batch_size"]
        )
    )

    dense_rows = surface_df[
        surface_df["r"] == 0
    ].copy()

    capacity = float(
        (
            1000.0
            * dense_rows["batch_size"]
            / dense_rows["latency_median_ms"]
        ).max()
    )

    dense_b1_p95 = float(
        dense_rows[
            dense_rows["batch_size"] == 1
        ]["latency_p95_ms"].iloc[0]
    )

    print(
        "Dense capacity:",
        capacity,
        "req/s",
    )

    print(
        "Dense B=1 p95:",
        dense_b1_p95,
        "ms",
    )

    policies = [
        "dense",
        "static_pomt",
        "decoupled",
        "joint_full",
        "cliffserve",
    ]

    checkpoint_path = (
        outdir / "summary_checkpoint.csv"
    )

    existing_df, completed_keys = load_checkpoint(
        checkpoint_path
    )

    rows = (
        existing_df.to_dict("records")
        if not existing_df.empty
        else []
    )

    total_expected = (
        len(cfg["epsilons"])
        * len(cfg["seeds"])
        * 2
        * len(cfg["load_factors"])
        * len(cfg["deadline_factors"])
        * len(policies)
    )

    print(
        f"Expected scenarios: {total_expected}"
    )

    print(
        f"Already completed: {len(completed_keys)}"
    )

    for epsilon_pp in cfg["epsilons"]:

        fast_surface = FastSurface(
            surface_df,
            epsilon_pp=float(epsilon_pp),
            cliff_min_relative_gain=float(
                cfg[
                    "cliff_min_relative_gain"
                ]
            ),
            cliff_min_absolute_gain_ms=float(
                cfg[
                    "cliff_min_absolute_gain_ms"
                ]
            ),
        )

        static_r = choose_static_r(
            fast_surface
        )

        print()
        print(
            f"epsilon={epsilon_pp} pp, "
            f"static R={static_r}"
        )

        for seed in cfg["seeds"]:

            for pattern in [
                "poisson",
                "onoff",
            ]:

                for load_factor in cfg[
                    "load_factors"
                ]:

                    offered_qps = (
                        capacity
                        * float(load_factor)
                    )

                    for deadline_factor in cfg[
                        "deadline_factors"
                    ]:

                        deadline_ms = (
                            dense_b1_p95
                            * float(
                                deadline_factor
                            )
                        )

                        if pattern == "poisson":

                            requests = poisson_trace(
                                n=cfg[
                                    "requests_per_trace"
                                ],
                                qps=offered_qps,
                                deadline_ms=deadline_ms,
                                seed=seed,
                            )

                        else:

                            requests = onoff_trace(
                                n=cfg[
                                    "requests_per_trace"
                                ],
                                qps=offered_qps,
                                deadline_ms=deadline_ms,
                                seed=seed,
                            )

                        for policy in policies:

                            key = (
                                float(epsilon_pp),
                                int(seed),
                                str(pattern),
                                float(load_factor),
                                float(deadline_factor),
                                str(policy),
                            )

                            if key in completed_keys:
                                continue

                            scheduler = build_scheduler(
                                policy,
                                fast_surface,
                                max_batch,
                            )

                            sampler = LatencySampler(
                                args.latency_samples,
                                seed=(
                                    seed
                                    + 10000
                                ),
                            )

                            completed, batches = replay(
                                requests=requests,
                                scheduler=scheduler,
                                sampler=sampler,
                                batch_window_ms=float(
                                    cfg[
                                        "batch_window_ms"
                                    ]
                                ),
                            )

                            result = summarize_replay(
                                completed,
                                batches,
                            )

                            result.update(
                                {
                                    "epsilon_pp":
                                        float(
                                            epsilon_pp
                                        ),

                                    "seed":
                                        int(seed),

                                    "pattern":
                                        pattern,

                                    "load_factor":
                                        float(
                                            load_factor
                                        ),

                                    "offered_qps":
                                        offered_qps,

                                    "deadline_factor":
                                        float(
                                            deadline_factor
                                        ),

                                    "deadline_ms":
                                        deadline_ms,

                                    "policy":
                                        policy,

                                    "static_r":
                                        static_r,
                                }
                            )

                            rows.append(result)

                            append_checkpoint(
                                checkpoint_path,
                                result,
                            )

                            completed_keys.add(
                                key
                            )

                            print(
                                f"eps={epsilon_pp} "
                                f"s={seed} "
                                f"{pattern} "
                                f"L={load_factor} "
                                f"D={deadline_factor} "
                                f"{policy}: "
                                f"SLO={result['slo_attainment']:.3f}, "
                                f"QPS={result['throughput_qps']:.1f}, "
                                f"OH={result['scheduler_overhead_mean_us']:.2f}us"
                            )

    raw = pd.DataFrame(rows)

    raw.to_csv(
        outdir / "summary_raw.csv",
        index=False,
    )

    # ========================================================
    # Aggregate seeds
    # ========================================================

    group_keys = [
        "epsilon_pp",
        "pattern",
        "load_factor",
        "deadline_factor",
        "policy",
    ]

    aggregate = (
        raw.groupby(
            group_keys,
            as_index=False,
        )
        .agg(
            slo_mean=(
                "slo_attainment",
                "mean",
            ),

            slo_std=(
                "slo_attainment",
                "std",
            ),

            throughput_mean=(
                "throughput_qps",
                "mean",
            ),

            throughput_std=(
                "throughput_qps",
                "std",
            ),

            p95_latency_mean=(
                "e2e_p95_ms",
                "mean",
            ),

            p99_latency_mean=(
                "e2e_p99_ms",
                "mean",
            ),

            scheduler_overhead_mean_us=(
                "scheduler_overhead_mean_us",
                "mean",
            ),

            scheduler_overhead_p95_us=(
                "scheduler_overhead_p95_us",
                "mean",
            ),

            mean_batch_size=(
                "mean_batch_size",
                "mean",
            ),

            mean_r=(
                "mean_r",
                "mean",
            ),
        )
    )

    aggregate.to_csv(
        outdir / "summary_aggregate.csv",
        index=False,
    )

    # ========================================================
    # Joint / CliffServe vs decoupled
    # ========================================================

    compare_keys = [
        "epsilon_pp",
        "pattern",
        "load_factor",
        "deadline_factor",
    ]

    dec = aggregate[
        aggregate["policy"]
        == "decoupled"
    ][
        compare_keys
        + [
            "slo_mean",
            "throughput_mean",
        ]
    ].rename(
        columns={
            "slo_mean":
                "decoupled_slo",

            "throughput_mean":
                "decoupled_throughput",
        }
    )

    ours = aggregate[
        aggregate["policy"]
        == "cliffserve"
    ][
        compare_keys
        + [
            "slo_mean",
            "throughput_mean",
        ]
    ].rename(
        columns={
            "slo_mean":
                "cliffserve_slo",

            "throughput_mean":
                "cliffserve_throughput",
        }
    )

    comparison = dec.merge(
        ours,
        on=compare_keys,
        validate="one_to_one",
    )

    comparison[
        "slo_gain_pp"
    ] = (
        100.0
        * (
            comparison[
                "cliffserve_slo"
            ]
            - comparison[
                "decoupled_slo"
            ]
        )
    )

    comparison[
        "throughput_gain_pct"
    ] = (
        100.0
        * (
            comparison[
                "cliffserve_throughput"
            ]
            / comparison[
                "decoupled_throughput"
            ]
            - 1.0
        )
    )

    comparison.to_csv(
        outdir
        / "cliffserve_vs_decoupled.csv",
        index=False,
    )

    # ========================================================
    # Best differences
    # ========================================================

    best = comparison.sort_values(
        [
            "epsilon_pp",
            "slo_gain_pp",
        ],
        ascending=[
            True,
            False,
        ],
    )

    best.to_csv(
        outdir
        / "cliffserve_vs_decoupled_sorted.csv",
        index=False,
    )

    # ========================================================
    # Scheduler overhead summary
    # ========================================================

    overhead = (
        aggregate.groupby(
            [
                "epsilon_pp",
                "policy",
            ],
            as_index=False,
        )
        .agg(
            mean_overhead_us=(
                "scheduler_overhead_mean_us",
                "mean",
            ),

            mean_p95_overhead_us=(
                "scheduler_overhead_p95_us",
                "mean",
            ),
        )
    )

    overhead.to_csv(
        outdir
        / "scheduler_overhead.csv",
        index=False,
    )

    print()
    print("=" * 80)
    print("FINAL SCHEDULER EXPERIMENT COMPLETE")
    print("=" * 80)

    print()
    print("Scheduler overhead:")
    print(
        overhead.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
