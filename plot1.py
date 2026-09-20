import matplotlib.pyplot as plt
import numpy as np

benchmarks = ["B1", "B2", "B3", "B4", "B5"]

train_hs  = [99, 174, 909, 95, 26]
train_nhs = [340, 5285, 4643, 4452, 2716]

test_hs  = [226, 498, 1808, 177, 41]
test_nhs = [4679, 41298, 46333, 31890, 19327]

x = np.arange(len(benchmarks))
width = 0.36

fig, axes = plt.subplots(
    1, 2,
    figsize=(12, 5.5),
    sharey=True
)

# -------------------------
# Training set
# -------------------------
axes[0].bar(
    x - width/2,
    train_hs,
    width,
    label="Hotspot (HS)"
)

axes[0].bar(
    x + width/2,
    train_nhs,
    width,
    label="Non-hotspot (NHS)"
)

axes[0].set_title("(a) Training set", fontsize=13, fontweight="bold")
axes[0].set_xticks(x)
axes[0].set_xticklabels(benchmarks)
axes[0].set_yscale("log")
axes[0].set_ylabel("Number of layout patterns")
axes[0].grid(axis="y", linestyle="--", alpha=0.35)

# -------------------------
# Test set
# -------------------------
axes[1].bar(
    x - width/2,
    test_hs,
    width,
    label="Hotspot (HS)"
)

axes[1].bar(
    x + width/2,
    test_nhs,
    width,
    label="Non-hotspot (NHS)"
)

axes[1].set_title("(b) Official test set", fontsize=13, fontweight="bold")
axes[1].set_xticks(x)
axes[1].set_xticklabels(benchmarks)
axes[1].set_yscale("log")
axes[1].grid(axis="y", linestyle="--", alpha=0.35)

# Common formatting
for ax in axes:
    ax.set_xlabel("ICCAD-12 benchmark")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=2,
    bbox_to_anchor=(0.5, 1.02),
    frameon=False
)

fig.suptitle(
    "Severe class imbalance across ICCAD-12 lithography hotspot benchmarks",
    fontsize=15,
    fontweight="bold",
    y=1.08
)

plt.tight_layout()

plt.savefig(
    "figure1_iccad_class_imbalance.png",
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    "figure1_iccad_class_imbalance.pdf",
    bbox_inches="tight"
)

plt.show()
