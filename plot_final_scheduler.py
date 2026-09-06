import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


ROOT = Path(
    "results/final_scheduler"
)

OUT = ROOT / "figures"
OUT.mkdir(
    parents=True,
    exist_ok=True,
)

df = pd.read_csv(
    ROOT
    / "summary_aggregate.csv"
)


# ============================================================
# SLO vs load
# ============================================================

for eps in sorted(
    df["epsilon_pp"].unique()
):

    data = df[
        (
            df["epsilon_pp"] == eps
        )
        & (
            df["pattern"]
            == "poisson"
        )
        & (
            df["deadline_factor"]
            == 8
        )
    ]

    fig, ax = plt.subplots(
        figsize=(
            7.2,
            4.8,
        )
    )

    for policy, group in data.groupby(
        "policy"
    ):

        group = group.sort_values(
            "load_factor"
        )

        ax.plot(
            group["load_factor"],
            100.0 * group["slo_mean"],
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

    fig.savefig(
        OUT
        / f"slo_vs_load_eps{eps:g}.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        OUT
        / f"slo_vs_load_eps{eps:g}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Scheduler overhead
# ============================================================

overhead = pd.read_csv(
    ROOT
    / "scheduler_overhead.csv"
)

for eps in sorted(
    overhead["epsilon_pp"].unique()
):

    data = overhead[
        overhead["epsilon_pp"] == eps
    ]

    fig, ax = plt.subplots(
        figsize=(
            7.0,
            4.5,
        )
    )

    ax.bar(
        data["policy"],
        data["mean_overhead_us"],
    )

    ax.set_ylabel(
        "Mean scheduling overhead (us)"
    )

    ax.tick_params(
        axis="x",
        rotation=25,
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        OUT
        / f"scheduler_overhead_eps{eps:g}.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        OUT
        / f"scheduler_overhead_eps{eps:g}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


print(
    "Saved figures to:",
    OUT,
)
