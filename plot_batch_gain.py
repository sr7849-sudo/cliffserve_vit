import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

INPUT = Path(
    "results/combined_latency/analysis/"
    "best_latency_configuration_by_batch.csv"
)

OUTPUT = Path(
    "results/combined_latency/analysis/"
    "fig_batch_vs_pruning_gain"
)

df = pd.read_csv(INPUT)

df = df.sort_values("batch_size")

x = df["batch_size"]
y = df["best_throughput_gain_pct"]

fig, ax = plt.subplots(
    figsize=(7.2, 4.8)
)

ax.plot(
    x,
    y,
    marker="o",
    linewidth=2,
)

ax.axhline(
    0,
    linewidth=1,
)

ax.set_xlabel(
    "Batch size",
    fontsize=12,
)

ax.set_ylabel(
    "Maximum throughput gain from token pruning (%)",
    fontsize=12,
)

ax.set_xticks(x)

ax.grid(
    alpha=0.25
)

for batch, gain in zip(x, y):
    ax.annotate(
        f"{gain:.1f}%",
        xy=(batch, gain),
        xytext=(0, 7),
        textcoords="offset points",
        ha="center",
        fontsize=9,
    )

fig.tight_layout()

fig.savefig(
    str(OUTPUT) + ".pdf",
    bbox_inches="tight",
)

fig.savefig(
    str(OUTPUT) + ".png",
    dpi=300,
    bbox_inches="tight",
)

print("Saved:")
print(str(OUTPUT) + ".pdf")
print(str(OUTPUT) + ".png")
