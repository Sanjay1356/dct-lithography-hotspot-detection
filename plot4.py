import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Figure 4: B3 Frequency-Band Ablation
# ============================================================

methods = [
    "Spatial",
    "Full DCT",
    "Multi-band DCT",
    "LOW",
    "MID",
    "HIGH"
]

# B3 ablation results (%)
balanced_accuracy = [
    93.55,
    50.18,
    86.55,
    50.00,
    67.10,
    50.60
]

recall = [
    98.34,
    100.00,
    96.24,
    0.00,
    98.34,
    100.00
]

specificity = [
    88.77,
    0.36,
    76.86,
    100.00,
    35.87,
    1.19
]

x = np.arange(len(methods))
width = 0.34

# ============================================================
# Figure
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(10, 4.8)
)

# ============================================================
# (a) Balanced Accuracy
# ============================================================

bars_ba = axes[0].bar(
    x,
    balanced_accuracy,
    width=0.62
)

axes[0].set_title(
    "(a) Balanced accuracy",
    fontsize=11,
    fontweight="bold",
    pad=10
)

axes[0].set_ylabel(
    "Balanced accuracy (%)",
    fontsize=10
)

axes[0].set_xticks(x)
axes[0].set_xticklabels(
    methods,
    rotation=25,
    ha="right",
    fontsize=8.5
)

axes[0].set_ylim(0, 105)
axes[0].set_yticks(np.arange(0, 101, 20))

axes[0].grid(
    axis="y",
    linestyle="--",
    linewidth=0.7,
    alpha=0.35
)

axes[0].set_axisbelow(True)

axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

for bar in bars_ba:

    height = bar.get_height()

    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        height + 1.2,
        f"{height:.1f}",
        ha="center",
        va="bottom",
        fontsize=7.5
    )


# ============================================================
# (b) Recall vs Specificity
# ============================================================

bars_recall = axes[1].bar(
    x - width / 2,
    recall,
    width,
    label="Recall"
)

bars_specificity = axes[1].bar(
    x + width / 2,
    specificity,
    width,
    label="Specificity"
)

axes[1].set_title(
    "(b) Recall vs. specificity",
    fontsize=11,
    fontweight="bold",
    pad=10
)

axes[1].set_ylabel(
    "Score (%)",
    fontsize=10
)

axes[1].set_xticks(x)
axes[1].set_xticklabels(
    methods,
    rotation=25,
    ha="right",
    fontsize=8.5
)

axes[1].set_ylim(0, 105)
axes[1].set_yticks(np.arange(0, 101, 20))

axes[1].grid(
    axis="y",
    linestyle="--",
    linewidth=0.7,
    alpha=0.35
)

axes[1].set_axisbelow(True)

axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

for bars in [bars_recall, bars_specificity]:

    for bar in bars:

        height = bar.get_height()

        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            height + 1.2,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=7
        )


# ============================================================
# Shared Legend
# ============================================================

handles, labels = axes[1].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=2,
    frameon=False,
    fontsize=8.5
)

# ============================================================
# Layout
# ============================================================

plt.tight_layout(
    rect=[0, 0, 1, 0.94]
)

# ============================================================
# Save
# ============================================================

plt.savefig(
    "figure4_b3_frequency_ablation.png",
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    "figure4_b3_frequency_ablation.pdf",
    bbox_inches="tight"
)

plt.show()
