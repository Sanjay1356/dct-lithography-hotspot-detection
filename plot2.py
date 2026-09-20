import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Figure 2: Balanced Accuracy across ICCAD-12 benchmarks
# ============================================================

benchmarks = ["B1", "B2", "B3", "B4", "B5"]

# Balanced Accuracy (%)
r0 = [95.02, 97.05, 93.55, 82.84, 93.82]
r1 = [50.00, 52.10, 50.18, 55.42, 49.99]
r2 = [50.00, 79.80, 86.55, 50.01, 53.66]
r3 = [92.42, 50.00, 53.84, 49.99, 76.82]

x = np.arange(len(benchmarks))
width = 0.19

# ------------------------------------------------------------
# Figure
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5.2))

# ------------------------------------------------------------
# Bars
# ------------------------------------------------------------

bars0 = ax.bar(
    x - 1.5 * width,
    r0,
    width,
    label="R0: Spatial CNN"
)

bars1 = ax.bar(
    x - 0.5 * width,
    r1,
    width,
    label="R1: Full DCT"
)

bars2 = ax.bar(
    x + 0.5 * width,
    r2,
    width,
    label="R2: Multi-band DCT"
)

bars3 = ax.bar(
    x + 1.5 * width,
    r3,
    width,
    label="R3: Spatial + DCT"
)

# ------------------------------------------------------------
# Axes
# ------------------------------------------------------------

ax.set_xlabel(
    "ICCAD-12 benchmark",
    fontsize=10
)

ax.set_ylabel(
    "Balanced accuracy (%)",
    fontsize=10
)

ax.set_xticks(x)
ax.set_xticklabels(benchmarks)

ax.set_ylim(0, 105)
ax.set_yticks(np.arange(0, 101, 20))

# ------------------------------------------------------------
# Grid
# ------------------------------------------------------------

ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.7,
    alpha=0.35
)

ax.set_axisbelow(True)

# Remove top/right borders
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ------------------------------------------------------------
# Value labels
# ------------------------------------------------------------

for bars in [bars0, bars1, bars2, bars3]:
    for bar in bars:

        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1.0,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=7
        )

# ------------------------------------------------------------
# Legend
# ------------------------------------------------------------

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.08),
    ncol=2,
    frameon=False,
    fontsize=8.5
)

# ------------------------------------------------------------
# Layout
# ------------------------------------------------------------

plt.tight_layout()

# ------------------------------------------------------------
# Save high-quality outputs
# ------------------------------------------------------------

plt.savefig(
    "figure2_balanced_accuracy.png",
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    "figure2_balanced_accuracy.pdf",
    bbox_inches="tight"
)

plt.show()
