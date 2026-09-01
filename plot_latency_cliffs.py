import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

INPUT = Path(
    "results/combined_latency/latency.csv"
)

OUTPUT = Path(
    "results/combined_latency/analysis/"
    "fig_latency_cliffs_by_batch"
)

df = pd.read_csv(INPUT)

fig, ax = plt.subplots(
    figsize=(8.0, 5.2)
)

for batch_size, group in df.groupby("batch_size"):
    group = group.sort_values("r")

    ax.plot(
        group["r"],
        group["latency_p95_ms"],
        marker="o",
        label=f"B={batch_size}",
    )

ax.set_xlabel(
    "Pruned tokens $R$",
    fontsize=12,
)

ax.set_ylabel(
    "p95 GPU latency (ms)",
    fontsize=12,
)

ax.grid(
    alpha=0.25
)

ax.legend(
    ncol=3,
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
