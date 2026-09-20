import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.fft import dct

BASE = os.path.expanduser(
    "~/Downloads/iccad-official/iccad1"
)

samples = [
    (
        os.path.join(BASE, "train/train_hs"),
        "HS"
    ),
    (
        os.path.join(BASE, "train/train_nhs"),
        "NHS"
    )
]


def make_dct(path):

    image = Image.open(path).convert("L")

    image = image.resize((224, 224))

    image = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    # TRUE 2-D DCT
    d = dct(
        dct(
            image,
            axis=0,
            norm="ortho"
        ),
        axis=1,
        norm="ortho"
    )

    magnitude = np.abs(d)

    log_dct = np.log1p(magnitude)

    log_dct -= log_dct.min()

    if log_dct.max() > 0:
        log_dct /= log_dct.max()

    return image, d, log_dct


for directory, label in samples:

    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(".png")
    ]

    path = files[0]

    image, d, log_dct = make_dct(path)

    print("\n==============================")
    print(label)
    print(path)
    print("==============================")

    print(
        "Original:",
        "min =", image.min(),
        "max =", image.max(),
        "mean =", image.mean()
    )

    print(
        "DCT:",
        "min =", d.min(),
        "max =", d.max(),
        "mean =", d.mean()
    )

    print(
        "DCT magnitude:",
        "min =", np.abs(d).min(),
        "max =", np.abs(d).max(),
        "mean =", np.abs(d).mean()
    )

    print(
        "Log DCT:",
        "min =", log_dct.min(),
        "max =", log_dct.max(),
        "mean =", log_dct.mean()
    )

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap="gray")
    plt.title(label + " Spatial")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(
        np.log1p(np.abs(d)),
        cmap="gray"
    )
    plt.title(label + " DCT")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(
        log_dct,
        cmap="gray"
    )
    plt.title(label + " Normalized DCT")
    plt.axis("off")

    plt.tight_layout()

    output = os.path.expanduser(
        f"~/Downloads/{label}_dct_inspection.png"
    )

    plt.savefig(
        output,
        dpi=150,
        bbox_inches="tight"
    )

    print("Saved:", output)
