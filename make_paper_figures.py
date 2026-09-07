#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path("results")
OUT = ROOT / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)

QUALITY = {
    "DeiT-S": ROOT / "final_quality_surface" / "budgets" /
              "best_configs_all_epsilons.csv",

    "DeiT-B": ROOT / "deit_base_final" / "budgets" /
              "best_configs_all_epsilons.csv",
}

SURFACE = {
    "DeiT-S": ROOT / "final_quality_surface" /
              "quality_latency_surface.csv",

    "DeiT-B": ROOT / "deit_base_final" /
              "quality_latency_surface.csv",
}

SCHED = {
    "DeiT-S": ROOT / "final_scheduler" /
              "summary_aggregate.csv",

    "DeiT-B": ROOT / "deit_base_scheduler" /
              "summary_aggregate.csv",
}

OVERHEAD = {
    "DeiT-S": ROOT / "final_scheduler" /
              "scheduler_overhead.csv",

    "DeiT-B": ROOT / "deit_base_scheduler" /
              "scheduler_overhead.csv",
}


# ============================================================
# PUBLICATION SETTINGS
# ============================================================

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.6,
    "lines.markersize": 5,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
})

BATCHES = [1, 2, 4, 8, 16, 24, 32, 48, 64]

MODEL_STYLES = {
    ("DeiT-S", 1.0): ("o", "-"),
    ("DeiT-S", 2.0): ("s", "--"),
    ("DeiT-B", 1.0): ("^", "-."),
    ("DeiT-B", 2.0): ("D", ":"),
}


def save(fig, stem):
    fig.savefig(
        OUT / f"{stem}.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        OUT / f"{stem}.png",
        bbox_inches="tight",
        dpi=300,
    )

    plt.close(fig)


# ============================================================
# FIGURE 1
# QUALITY-CONSTRAINED SERVICE-RATE GAIN
# ============================================================

def plot_quality_gain():

    fig, ax = plt.subplots(
        figsize=(3.45, 2.55)
    )

    for model, path in QUALITY.items():

        x = pd.read_csv(path)

        for eps in [1.0, 2.0]:

            y = (
                x[x["epsilon_pp"] == eps]
                .sort_values("batch_size")
            )

            marker, linestyle = MODEL_STYLES[
                (model, eps)
            ]

            ax.plot(
                y["batch_size"],
                y["throughput_gain_pct"],
                marker=marker,
                linestyle=linestyle,
                label=f"{model}, $\\epsilon={eps:g}$ pp",
            )

    ax.axhline(
        0,
        linewidth=0.8,
    )

    ax.set_xlabel("Batch size")
    ax.set_ylabel("p95 service-rate gain (%)")

    ax.set_xticks(BATCHES)
    ax.set_xticklabels(
        [str(x) for x in BATCHES],
        rotation=35,
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.legend(
        frameon=False,
        ncol=2,
    )

    fig.tight_layout()

    save(
        fig,
        "fig1_quality_gain_two_models",
    )


# ============================================================
# FIGURE 2
# ACCURACY DROP VS PRUNING
#
# Shows A(B,R) ~ A(R)
# ============================================================

def plot_accuracy_drop():

    fig, ax = plt.subplots(
        figsize=(3.45, 2.55)
    )

    for model, path in SURFACE.items():

        x = pd.read_csv(path)

        stats = (
            x.groupby("r")["accuracy_drop_pp"]
            .agg(["mean", "min", "max"])
            .reset_index()
            .sort_values("r")
        )

        ax.plot(
            stats["r"],
            stats["mean"],
            marker="o",
            label=model,
        )

        ax.fill_between(
            stats["r"],
            stats["min"],
            stats["max"],
            alpha=0.15,
        )

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=0.9,
    )

    ax.axhline(
        2.0,
        linestyle=":",
        linewidth=0.9,
    )

    ax.set_xlabel("Tokens removed, $R$")
    ax.set_ylabel("Top-1 accuracy loss (pp)")

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.legend(
        frameon=False,
    )

    fig.tight_layout()

    save(
        fig,
        "fig2_accuracy_drop_vs_pruning",
    )


# ============================================================
# FIGURE 3
# LATENCY CLIFFS
#
# Representative batches only.
# Generate one clean figure per model.
# ============================================================

def plot_latency_cliffs():

    representative_batches = [
        8,
        16,
        32,
        64,
    ]

    marker_map = {
        8: "o",
        16: "s",
        32: "^",
        64: "D",
    }

    for model, path in SURFACE.items():

        x = pd.read_csv(path)

        fig, ax = plt.subplots(
            figsize=(3.45, 2.55)
        )

        for b in representative_batches:

            y = (
                x[x["batch_size"] == b]
                .sort_values("r")
            )

            ax.plot(
                y["r"],
                y["latency_p95_ms"],
                marker=marker_map[b],
                label=f"$B={b}$",
            )

        ax.set_xlabel("Tokens removed, $R$")
        ax.set_ylabel("p95 GPU latency (ms)")

        ax.grid(
            axis="y",
            alpha=0.25,
        )

        ax.legend(
            frameon=False,
            ncol=2,
        )

        fig.tight_layout()

        stem = (
            "fig3_latency_cliffs_"
            + model.lower().replace("-", "")
        )

        save(
            fig,
            stem,
        )


# ============================================================
# FIGURE 4
# COMPLETE WORKLOAD HEATMAP
#
# CliffServe - Decoupled SLO
# epsilon = 2 pp
# ============================================================

def plot_slo_heatmaps():

    EPS = 2.0

    for model, path in SCHED.items():

        x = pd.read_csv(path)

        keys = [
            "epsilon_pp",
            "pattern",
            "load_factor",
            "deadline_factor",
        ]

        cliff = (
            x[x["policy"] == "cliffserve"]
            .set_index(keys)
            [["slo_mean"]]
            .rename(
                columns={
                    "slo_mean": "cliff_slo"
                }
            )
        )

        dec = (
            x[x["policy"] == "decoupled"]
            .set_index(keys)
            [["slo_mean"]]
            .rename(
                columns={
                    "slo_mean": "dec_slo"
                }
            )
        )

        z = cliff.join(dec)

        z["gain_pp"] = (
            100.0
            * (
                z["cliff_slo"]
                - z["dec_slo"]
            )
        )

        z = z.reset_index()
        z = z[
            z["epsilon_pp"] == EPS
        ]

        for pattern in [
            "poisson",
            "onoff",
        ]:

            y = z[
                z["pattern"] == pattern
            ]

            table = (
                y.pivot(
                    index="load_factor",
                    columns="deadline_factor",
                    values="gain_pp",
                )
                .sort_index()
            )

            fig, ax = plt.subplots(
                figsize=(3.45, 2.75)
            )

            im = ax.imshow(
                table.values,
                origin="lower",
                aspect="auto",
            )

            ax.set_xticks(
                np.arange(
                    len(table.columns)
                )
            )

            ax.set_xticklabels(
                [
                    f"{v:g}"
                    for v in table.columns
                ]
            )

            ax.set_yticks(
                np.arange(
                    len(table.index)
                )
            )

            ax.set_yticklabels(
                [
                    f"{v:g}"
                    for v in table.index
                ]
            )

            ax.set_xlabel("Deadline factor")
            ax.set_ylabel(
                "Offered load / dense capacity"
            )

            for i in range(
                table.shape[0]
            ):
                for j in range(
                    table.shape[1]
                ):

                    val = table.iloc[i, j]

                    ax.text(
                        j,
                        i,
                        f"{val:.1f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                    )

            cbar = fig.colorbar(
                im,
                ax=ax,
                fraction=0.047,
                pad=0.03,
            )

            cbar.set_label(
                "$\\Delta$SLO vs. Decoupled (pp)"
            )

            fig.tight_layout()

            stem = (
                "fig4_slo_heatmap_"
                + model.lower().replace("-", "")
                + "_"
                + pattern
                + "_eps2"
            )

            save(
                fig,
                stem,
            )


# ============================================================
# FIGURE 5
# SLO VS LOAD
#
# Means and std across seeds.
#
# Generate for all deadline factors.
# We can choose the best representative one later
# rather than cherry-picking now.
# ============================================================

def plot_slo_vs_load():

    EPS = 2.0

    policies = [
        "dense",
        "static_pomt",
        "decoupled",
        "joint_full",
        "cliffserve",
    ]

    markers = {
        "dense": "o",
        "static_pomt": "s",
        "decoupled": "^",
        "joint_full": "D",
        "cliffserve": "P",
    }

    labels = {
        "dense": "Dense",
        "static_pomt": "Static-POMT",
        "decoupled": "Decoupled",
        "joint_full": "Joint-Full",
        "cliffserve": "CliffServe",
    }

    for model, path in SCHED.items():

        x = pd.read_csv(path)

        x = x[
            x["epsilon_pp"] == EPS
        ]

        for pattern in [
            "poisson",
            "onoff",
        ]:

            for deadline in [
                2,
                4,
                8,
                12,
            ]:

                y = x[
                    (x["pattern"] == pattern)
                    &
                    (
                        x["deadline_factor"]
                        == deadline
                    )
                ]

                fig, ax = plt.subplots(
                    figsize=(3.45, 2.65)
                )

                for policy in policies:

                    p = (
                        y[y["policy"] == policy]
                        .sort_values(
                            "load_factor"
                        )
                    )

                    ax.errorbar(
                        p["load_factor"],
                        100.0
                        * p["slo_mean"],
                        yerr=100.0
                        * p["slo_std"],
                        marker=markers[policy],
                        capsize=2,
                        label=labels[policy],
                    )

                ax.set_xlabel(
                    "Offered load / dense capacity"
                )

                ax.set_ylabel(
                    "SLO attainment (%)"
                )

                ax.set_ylim(
                    -2,
                    102,
                )

                ax.grid(
                    axis="y",
                    alpha=0.25,
                )

                ax.legend(
                    frameon=False,
                    fontsize=7,
                )

                fig.tight_layout()

                stem = (
                    "fig5_slo_"
                    + model.lower().replace("-", "")
                    + "_"
                    + pattern
                    + f"_d{deadline}_eps2"
                )

                save(
                    fig,
                    stem,
                )


# ============================================================
# FIGURE 6
# SCHEDULER OVERHEAD
# epsilon = 2 pp
# ============================================================

def plot_scheduler_overhead():

    rows = []

    for model, path in OVERHEAD.items():

        x = pd.read_csv(path)

        x = x[
            x["epsilon_pp"] == 2.0
        ].copy()

        x["model"] = model

        rows.append(x)

    z = pd.concat(
        rows,
        ignore_index=True,
    )

    policies = [
        "dense",
        "static_pomt",
        "decoupled",
        "cliffserve",
        "joint_full",
    ]

    labels = [
        "Dense",
        "Static",
        "Decoup.",
        "Cliff",
        "Joint",
    ]

    fig, ax = plt.subplots(
        figsize=(3.45, 2.55)
    )

    width = 0.36
    positions = np.arange(
        len(policies)
    )

    for offset, model in [
        (-width / 2, "DeiT-S"),
        (+width / 2, "DeiT-B"),
    ]:

        y = (
            z[z["model"] == model]
            .set_index("policy")
            .loc[policies]
        )

        ax.bar(
            positions + offset,
            y["mean_overhead_us"],
            width=width,
            label=model,
        )

    ax.set_xticks(
        positions
    )

    ax.set_xticklabels(
        labels,
        rotation=25,
    )

    ax.set_ylabel(
        "Mean decision time ($\\mu$s)"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.legend(
        frameon=False,
    )

    fig.tight_layout()

    save(
        fig,
        "fig6_scheduler_overhead_eps2",
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("Generating paper figures...")

    plot_quality_gain()
    print("  Figure 1 done")

    plot_accuracy_drop()
    print("  Figure 2 done")

    plot_latency_cliffs()
    print("  Figure 3 done")

    plot_slo_heatmaps()
    print("  Figure 4 done")

    plot_slo_vs_load()
    print("  Figure 5 done")

    plot_scheduler_overhead()
    print("  Figure 6 done")

    print()
    print(
        "Saved all figures to:",
        OUT,
    )
