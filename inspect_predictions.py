import os
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from PIL import Image
from scipy.fft import dct


# ============================================================
# B1
# ============================================================

ROOT = os.path.expanduser(
    "~/Downloads/iccad-official/iccad1"
)

TRAIN = os.path.join(ROOT, "train")
TEST = os.path.join(ROOT, "test")

MODEL_PATH = os.path.expanduser(
    "~/Downloads/litho_DCTv2_B1/R1v2_DCT_B1.keras"
)

IMG_SIZE = (224, 224)


def pngs(directory):

    return sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(".png")
    ])


def collect(directory):

    hs = pngs(
        os.path.join(directory, "test_hs")
    )

    nhs = pngs(
        os.path.join(directory, "test_nhs")
    )

    paths = nhs + hs
    labels = (
        [0] * len(nhs) +
        [1] * len(hs)
    )

    return paths, np.array(labels)


test_paths, test_labels = collect(TEST)


# ============================================================
# LOAD MODEL
# ============================================================

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("\nModel loaded.")
print("Parameters:", model.count_params())


# ============================================================
# DCT
# ============================================================

def make_input(path, scale):

    image = Image.open(path).convert("L")

    image = image.resize(
        IMG_SIZE,
        Image.Resampling.BILINEAR
    )

    image = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    coeff = dct(
        dct(
            image,
            axis=0,
            type=2,
            norm="ortho"
        ),
        axis=1,
        type=2,
        norm="ortho"
    )

    transformed = (
        np.sign(coeff) *
        np.log1p(np.abs(coeff))
    )

    transformed = (
        transformed / scale
    )

    transformed = (
        transformed + 1.0
    ) / 2.0

    transformed = np.clip(
        transformed,
        0.0,
        1.0
    )

    transformed = np.repeat(
        transformed[:, :, None],
        3,
        axis=2
    )

    return transformed.astype(
        np.float32
    )


# ============================================================
# RECOVER SCALE FROM TRAINING DATA
# ============================================================

print("\nRecovering training-set scale...")

train_hs = pngs(
    os.path.join(TRAIN, "train_hs")
)

train_nhs = pngs(
    os.path.join(TRAIN, "train_nhs")
)

train_paths = (
    train_hs +
    train_nhs
)

scale = 0.0

for path in train_paths:

    image = Image.open(path).convert("L")

    image = image.resize(
        IMG_SIZE,
        Image.Resampling.BILINEAR
    )

    image = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    coeff = dct(
        dct(
            image,
            axis=0,
            type=2,
            norm="ortho"
        ),
        axis=1,
        type=2,
        norm="ortho"
    )

    transformed = (
        np.sign(coeff) *
        np.log1p(np.abs(coeff))
    )

    scale = max(
        scale,
        np.max(np.abs(transformed))
    )

print(
    "Scale:",
    scale
)


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

print("\nComputing predictions...")

# Take all HS and a comparable NHS subset
sample_nhs = test_paths[
    :min(226, len(test_paths))
]

# Find actual HS
hs_paths = [
    p for p, y in zip(
        test_paths,
        test_labels
    )
    if y == 1
]

sample_paths = (
    sample_nhs +
    hs_paths
)

X = np.stack([
    make_input(
        p,
        scale
    )
    for p in sample_paths
])

y = np.array([
    0 if p in sample_nhs else 1
    for p in sample_paths
])

pred = model.predict(
    X,
    verbose=0
).flatten()


# ============================================================
# STATISTICS
# ============================================================

nhs_pred = pred[y == 0]
hs_pred = pred[y == 1]

print("\n========================================")
print("PREDICTION DISTRIBUTION")
print("========================================")

print("\nNHS predictions:")
print("count :", len(nhs_pred))
print("min   :", np.min(nhs_pred))
print("max   :", np.max(nhs_pred))
print("mean  :", np.mean(nhs_pred))
print("median:", np.median(nhs_pred))

print("\nHS predictions:")
print("count :", len(hs_pred))
print("min   :", np.min(hs_pred))
print("max   :", np.max(hs_pred))
print("mean  :", np.mean(hs_pred))
print("median:", np.median(hs_pred))

print("\nFirst 20 NHS:")
print(nhs_pred[:20])

print("\nFirst 20 HS:")
print(hs_pred[:20])
