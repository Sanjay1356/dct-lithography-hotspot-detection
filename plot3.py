import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Figure 3: Average performance across ICCAD-12 benchmarks
# ============================================================

metrics = [
    "Balanced Accuracy",
    "Precision",
    "Recall",
    "Specificity",
    "F1"
]

# Average values across B1-B5 (%)
r0 = [92.46, 48.64, 88.39, 96.52, 61.11]
r1 = [51.54, 14.00, 40.04, 62.23, 3.28]
r2 = [64.00, 23.50, 60.35, 67.66, 18.98]
r3 = [64.61, 32.78, 31.01, 98.22, 25.79]

x = np.arange(len(metrics))
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

ax.set_ylabel(
    "Average score (%)",
    fontsize=10
)

ax.set_xticks(x)
ax.set_xticklabels(
    metrics,
    fontsize=9
)

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

# Remove unnecessary borders
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
# Save
# ------------------------------------------------------------

plt.savefig(
    "figure3_average_metrics.png",
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    "figure3_average_metrics.pdf",
    bbox_inches="tight"
)

plt.show()
