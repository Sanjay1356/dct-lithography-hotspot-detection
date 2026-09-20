import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Figure 6: Feature-level discriminability using AUC
# ============================================================

benchmarks = ["B1", "B2", "B3", "B4", "B5"]

features = [
    "Spatial mean",
    "Spatial variance",
    "Low-frequency energy",
    "Mid-frequency energy",
    "High-frequency energy",
    "High/low energy ratio",
    "Spectral centroid",
    "DC energy ratio",
    "Total DCT energy"
]

# AUC values
#
# Rows = features
# Columns = B1 ... B5

auc = np.array([
    [0.7392, 0.8178, 0.6577, 0.6578, 0.6493],  # Spatial mean
    [0.7297, 0.8379, 0.6563, 0.6499, 0.6579],  # Spatial variance
    [0.6442, 0.5903, 0.5348, 0.5498, 0.5486],  # Low
    [0.6620, 0.6163, 0.6475, 0.6200, 0.6112],  # Mid
    [0.5456, 0.6038, 0.7904, 0.8356, 0.8308],  # High
    [0.5405, 0.5938, 0.7825, 0.8289, 0.8224],  # High/low
    [0.5805, 0.6032, 0.5845, 0.6161, 0.6456],  # Centroid
    [0.7411, 0.8165, 0.6590, 0.6617, 0.6511],  # DC
    [0.7366, 0.8192, 0.6565, 0.6538, 0.6485]   # Total
])

# ============================================================
# Figure
# ============================================================

fig, ax = plt.subplots(
    figsize=(9.5, 6.2)
)

im = ax.imshow(
    auc,
    aspect="auto",
    vmin=0.5,
    vmax=0.85
)

# ============================================================
# Axes
# ============================================================

ax.set_xticks(
    np.arange(len(benchmarks))
)

ax.set_xticklabels(
    benchmarks,
    fontsize=9
)

ax.set_yticks(
    np.arange(len(features))
)

ax.set_yticklabels(
    features,
    fontsize=9
)

ax.set_xlabel(
    "ICCAD-12 benchmark",
    fontsize=10
)

ax.set_ylabel(
    "Feature",
    fontsize=10
)

# ============================================================
# Annotate AUC values
# ============================================================

for i in range(len(features)):
    for j in range(len(benchmarks)):

        value = auc[i, j]

        # Use white text for stronger cells
        if value > 0.70:
            text_color = "white"
        else:
            text_color = "black"

        ax.text(
            j,
            i,
            f"{value:.3f}",
            ha="center",
            va="center",
            fontsize=8,
            color=text_color
        )

# ============================================================
# Colorbar
# ============================================================

cbar = fig.colorbar(
    im,
    ax=ax,
    fraction=0.035,
    pad=0.03
)

cbar.set_label(
    "AUC",
    fontsize=9
)

# ============================================================
# Grid
# ============================================================

ax.set_xticks(
    np.arange(-0.5, len(benchmarks), 1),
    minor=True
)

ax.set_yticks(
    np.arange(-0.5, len(features), 1),
    minor=True
)

ax.grid(
    which="minor",
    color="white",
    linewidth=1.5
)

ax.tick_params(
    which="minor",
    bottom=False,
    left=False
)

# ============================================================
# Title
# ============================================================

ax.set_title(
    "Feature-level discriminability between hotspot and non-hotspot patterns",
    fontsize=12,
    fontweight="bold",
    pad=12
)

# ============================================================
# Layout
# ============================================================

plt.tight_layout()

# ============================================================
# Save
# ============================================================

plt.savefig(
    "figure6_frequency_feature_auc.png",
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    "figure6_frequency_feature_auc.pdf",
    bbox_inches="tight"
)

plt.show()
