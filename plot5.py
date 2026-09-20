import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Figure 5: B3 Confusion Matrices
# ============================================================

methods = [
    "R0: Spatial CNN",
    "R1: Full DCT",
    "R2: Multi-band DCT",
    "R3: Spatial + DCT"
]

# Confusion matrix format:
# [[TN, FP],
#  [FN, TP]]

confusion_matrices = [
    np.array([
        [41128, 5205],
        [30, 1778]
    ]),

    np.array([
        [168, 46165],
        [0, 1808]
    ]),

    np.array([
        [35610, 10723],
        [68, 1740]
    ]),

    np.array([
        [46175, 158],
        [1663, 145]
    ])
]

class_labels = ["NHS", "HS"]

# ============================================================
# Figure
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(9.2, 7.2)
)

axes = axes.flatten()

# Same color scale across all matrices
global_max = max(
    matrix.max()
    for matrix in confusion_matrices
)

# ============================================================
# Plot confusion matrices
# ============================================================

for ax, matrix, method in zip(
    axes,
    confusion_matrices,
    methods
):

    im = ax.imshow(
        matrix,
        cmap="Blues",
        vmin=0,
        vmax=global_max
    )

    # --------------------------------------------------------
    # Axis labels
    # --------------------------------------------------------

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(
        class_labels,
        fontsize=9
    )

    ax.set_yticklabels(
        class_labels,
        fontsize=9
    )

    ax.set_xlabel(
        "Predicted class",
        fontsize=9
    )

    ax.set_ylabel(
        "Actual class",
        fontsize=9
    )

    ax.set_title(
        method,
        fontsize=11,
        fontweight="bold",
        pad=8
    )

    # --------------------------------------------------------
    # Cell annotations
    # --------------------------------------------------------

    for i in range(2):
        for j in range(2):

            value = matrix[i, j]

            row_total = matrix[i].sum()
            percentage = (value / row_total) * 100

            # Determine annotation color
            threshold = global_max * 0.45

            text_color = (
                "white"
                if value > threshold
                else "black"
            )

            ax.text(
                j,
                i,
                f"{value:,}\n({percentage:.1f}%)",
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
                fontweight="bold"
            )

    # --------------------------------------------------------
    # Cell boundaries
    # --------------------------------------------------------

    ax.set_xticks(
        np.arange(-0.5, 2, 1),
        minor=True
    )

    ax.set_yticks(
        np.arange(-0.5, 2, 1),
        minor=True
    )

    ax.grid(
        which="minor",
        color="white",
        linestyle="-",
        linewidth=2
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False
    )


# ============================================================
# Leave dedicated space on the right for colorbar
# ============================================================

fig.subplots_adjust(
    left=0.08,
    right=0.87,
    bottom=0.08,
    top=0.90,
    wspace=0.45,
    hspace=0.45
)

# ============================================================
# Dedicated colorbar axis
# ============================================================

cbar_ax = fig.add_axes([
    0.90,   # left
    0.23,   # bottom
    0.018,  # width
    0.54    # height
])

cbar = fig.colorbar(
    im,
    cax=cbar_ax
)

cbar.set_label(
    "Number of test samples",
    fontsize=9,
    labelpad=8
)

cbar.ax.tick_params(
    labelsize=8
)

# ============================================================
# Figure title
# ============================================================

fig.suptitle(
    "B3 confusion matrices for spatial and DCT-based representations",
    fontsize=13,
    fontweight="bold",
    y=0.96
)

# ============================================================
# Save
# ============================================================

plt.savefig(
    "figure5_b3_confusion_matrices.png",
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    "figure5_b3_confusion_matrices.pdf",
    bbox_inches="tight"
)

plt.show()
