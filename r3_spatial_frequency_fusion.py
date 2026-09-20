import os
import sys
import time
import random
import numpy as np

from PIL import Image
from scipy.fft import dctn

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Nadam

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
)

# ============================================================
# R3 SPATIAL + MULTI-BAND DCT FUSION
# ============================================================

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10

TARGET_PARAMS = 12873

DATA_ROOT = os.path.expanduser("~/Downloads/iccad-official")
CACHE_ROOT = os.path.expanduser("~/Downloads/litho_r3_fusion_cache")

# Same frequency cutoffs used by R2
LOW_CUTOFF = 0.15
MID_CUTOFF = 0.40

BENCHMARKS = {
    "B1": "iccad1",
    "B2": "iccad2",
    "B3": "iccad3",
    "B4": "iccad4",
    "B5": "iccad5",
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# PATHS
# ============================================================

def get_paths(benchmark):
    folder = BENCHMARKS[benchmark]

    root = os.path.join(DATA_ROOT, folder)

    train_hs = os.path.join(root, "train", "train_hs")
    train_nhs = os.path.join(root, "train", "train_nhs")

    test_hs = os.path.join(root, "test", "test_hs")
    test_nhs = os.path.join(root, "test", "test_nhs")

    return train_hs, train_nhs, test_hs, test_nhs


# ============================================================
# FILE DISCOVERY
# ============================================================

def get_files(folder):
    files = []

    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    for name in os.listdir(folder):
        lower = name.lower()

        if lower.endswith((".png", ".jpg", ".jpeg")):
            files.append(os.path.join(folder, name))

    files.sort()

    return files


# ============================================================
# DCT BAND MASKS
# ============================================================

def create_band_masks():

    y, x = np.meshgrid(
        np.arange(IMG_SIZE),
        np.arange(IMG_SIZE),
        indexing="ij"
    )

    # Normalized radial frequency.
    # DC is located at (0, 0).
    radial = np.sqrt(
        (y / (IMG_SIZE - 1)) ** 2 +
        (x / (IMG_SIZE - 1)) ** 2
    )

    # Normalize so maximum radial frequency is 1.
    radial = radial / radial.max()

    low_mask = radial <= LOW_CUTOFF

    mid_mask = (
        (radial > LOW_CUTOFF) &
        (radial <= MID_CUTOFF)
    )

    high_mask = radial > MID_CUTOFF

    return low_mask, mid_mask, high_mask


LOW_MASK, MID_MASK, HIGH_MASK = create_band_masks()


print()
print("=" * 70)
print("R3 SPATIAL + MULTI-BAND DCT FUSION")
print("=" * 70)
print()
print("Frequency bands:")
print(f"LOW  : r <= {LOW_CUTOFF}")
print(f"MID  : {LOW_CUTOFF} < r <= {MID_CUTOFF}")
print(f"HIGH : r > {MID_CUTOFF}")
print()

print("Band coefficients:")
print(f"LOW  : {LOW_MASK.sum():,}")
print(f"MID  : {MID_MASK.sum():,}")
print(f"HIGH : {HIGH_MASK.sum():,}")
print(f"TOTAL: {(LOW_MASK.sum() + MID_MASK.sum() + HIGH_MASK.sum()):,}")
print()


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(path):

    # --------------------------------------------------------
    # Common spatial preprocessing
    # --------------------------------------------------------

    img = Image.open(path).convert("L")

    img = img.resize(
        (IMG_SIZE, IMG_SIZE),
        Image.Resampling.BILINEAR
    )

    spatial = np.asarray(
        img,
        dtype=np.float32
    )

    spatial /= 255.0

    # Spatial branch expects 3 channels.
    spatial_3 = np.repeat(
        spatial[..., np.newaxis],
        3,
        axis=-1
    )

    # --------------------------------------------------------
    # DCT preprocessing
    # --------------------------------------------------------

    coeff = dctn(
        spatial,
        type=2,
        norm="ortho"
    )

    # Same representation as R2:
    # absolute magnitude
    coeff = np.abs(coeff)

    # Log compression
    coeff = np.log1p(coeff)

    # Per-image min-max normalization
    cmin = coeff.min()
    cmax = coeff.max()

    if cmax > cmin:
        coeff = (coeff - cmin) / (cmax - cmin)
    else:
        coeff = np.zeros_like(coeff)

    # --------------------------------------------------------
    # Multi-band DCT
    # --------------------------------------------------------

    low = np.where(LOW_MASK, coeff, 0.0)
    mid = np.where(MID_MASK, coeff, 0.0)
    high = np.where(HIGH_MASK, coeff, 0.0)

    frequency = np.stack(
        [low, mid, high],
        axis=-1
    )

    return (
        spatial_3.astype(np.float16),
        frequency.astype(np.float16)
    )


# ============================================================
# CACHE CREATION
# ============================================================

def create_cache(name, files, labels):

    os.makedirs(CACHE_ROOT, exist_ok=True)

    spatial_path = os.path.join(
        CACHE_ROOT,
        f"{name}_spatial.dat"
    )

    frequency_path = os.path.join(
        CACHE_ROOT,
        f"{name}_frequency.dat"
    )

    labels_path = os.path.join(
        CACHE_ROOT,
        f"{name}_labels.npy"
    )

    shape_spatial = (
        len(files),
        IMG_SIZE,
        IMG_SIZE,
        3
    )

    shape_frequency = (
        len(files),
        IMG_SIZE,
        IMG_SIZE,
        3
    )

    # Reuse cache only if all files exist.
    if (
        os.path.exists(spatial_path)
        and os.path.exists(frequency_path)
        and os.path.exists(labels_path)
    ):

        print(f"Using existing cache: {name}")

        return (
            spatial_path,
            frequency_path,
            labels_path,
            shape_spatial,
            shape_frequency
        )

    print()
    print(f"Creating R3 cache: {name}")
    print(f"Images: {len(files):,}")

    spatial_memmap = np.memmap(
        spatial_path,
        dtype=np.float16,
        mode="w+",
        shape=shape_spatial
    )

    frequency_memmap = np.memmap(
        frequency_path,
        dtype=np.float16,
        mode="w+",
        shape=shape_frequency
    )

    start = time.time()

    for i, path in enumerate(files):

        spatial, frequency = preprocess_image(path)

        spatial_memmap[i] = spatial
        frequency_memmap[i] = frequency

        if (
            (i + 1) % 500 == 0
            or i + 1 == len(files)
        ):

            elapsed = time.time() - start

            print(
                f"Processed {i + 1:,}/{len(files):,} "
                f"({100 * (i + 1) / len(files):.1f}%) "
                f"[{elapsed:.1f}s]",
                end="\r"
            )

    spatial_memmap.flush()
    frequency_memmap.flush()

    np.save(
        labels_path,
        np.asarray(labels, dtype=np.int8)
    )

    print()
    print(f"Cache written: {name}")

    del spatial_memmap
    del frequency_memmap

    return (
        spatial_path,
        frequency_path,
        labels_path,
        shape_spatial,
        shape_frequency
    )


# ============================================================
# DATASET PREPARATION
# ============================================================

def prepare_dataset(benchmark):

    train_hs, train_nhs, test_hs, test_nhs = get_paths(
        benchmark
    )

    # --------------------------------------------------------
    # Training files
    # --------------------------------------------------------

    hs_train = get_files(train_hs)
    nhs_train = get_files(train_nhs)

    train_files = nhs_train + hs_train

    train_labels = (
        [0] * len(nhs_train) +
        [1] * len(hs_train)
    )

    # --------------------------------------------------------
    # Test files
    # --------------------------------------------------------

    hs_test = get_files(test_hs)
    nhs_test = get_files(test_nhs)

    test_files = nhs_test + hs_test

    test_labels = (
        [0] * len(nhs_test) +
        [1] * len(hs_test)
    )

    # --------------------------------------------------------
    # Create caches
    # --------------------------------------------------------

    train_cache = create_cache(
        f"{benchmark}_train",
        train_files,
        train_labels
    )

    test_cache = create_cache(
        f"{benchmark}_test",
        test_files,
        test_labels
    )

    return train_cache, test_cache


# ============================================================
# DATA GENERATOR
# ============================================================

class FusionSequence(tf.keras.utils.Sequence):

    def __init__(
        self,
        spatial_path,
        frequency_path,
        labels_path,
        shape_spatial,
        shape_frequency,
        batch_size=32,
        shuffle=False
    ):

        self.spatial = np.memmap(
            spatial_path,
            dtype=np.float16,
            mode="r",
            shape=shape_spatial
        )

        self.frequency = np.memmap(
            frequency_path,
            dtype=np.float16,
            mode="r",
            shape=shape_frequency
        )

        self.labels = np.load(
            labels_path
        )

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.indices = np.arange(
            len(self.labels)
        )

        self.on_epoch_end()

    def __len__(self):

        return int(
            np.ceil(
                len(self.labels) /
                self.batch_size
            )
        )

    def __getitem__(self, index):

        start = index * self.batch_size

        end = min(
            start + self.batch_size,
            len(self.labels)
        )

        batch_indices = self.indices[start:end]

        spatial_batch = np.asarray(
            self.spatial[batch_indices],
            dtype=np.float32
        )

        frequency_batch = np.asarray(
            self.frequency[batch_indices],
            dtype=np.float32
        )

        labels_batch = self.labels[
            batch_indices
        ].astype(np.float32)

        return (
            {
                "spatial_input": spatial_batch,
                "frequency_input": frequency_batch
            },
            labels_batch
        )

    def on_epoch_end(self):

        if self.shuffle:

            rng = np.random.default_rng(SEED)

            rng.shuffle(
                self.indices
            )


# ============================================================
# SPATIAL BRANCH
# ============================================================

def spatial_branch():

    inputs = layers.Input(
        shape=(IMG_SIZE, IMG_SIZE, 3),
        name="spatial_input"
    )

    x = layers.Conv2D(
        10,
        (3, 3),
        activation="elu"
    )(inputs)

    x = layers.Conv2D(
        10,
        (3, 3),
        activation="elu"
    )(x)

    x = layers.Conv2D(
        10,
        (3, 3)
    )(x)

    x = layers.BatchNormalization(
        momentum=0.99,
        epsilon=0.001
    )(x)

    x = layers.ELU()(x)

    x = layers.MaxPooling2D(
        pool_size=(2, 2)
    )(x)

    x = layers.MaxPooling2D(
        pool_size=(5, 5)
    )(x)

    x = layers.Conv2D(
        10,
        (3, 3),
        activation="elu"
    )(x)

    x = layers.Conv2D(
        10,
        (3, 3),
        activation="elu"
    )(x)

    x = layers.Conv2D(
        10,
        (3, 3)
    )(x)

    x = layers.BatchNormalization(
        momentum=0.99,
        epsilon=0.001
    )(x)

    x = layers.ELU()(x)

    x = layers.MaxPooling2D(
        pool_size=(2, 2)
    )(x)

    return inputs, x


# ============================================================
# FREQUENCY BRANCH
# ============================================================

def frequency_branch():

    inputs = layers.Input(
        shape=(IMG_SIZE, IMG_SIZE, 3),
        name="frequency_input"
    )

    x = layers.Conv2D(
        5,
        (3, 3),
        activation="elu"
    )(inputs)

    x = layers.Conv2D(
        5,
        (3, 3),
        activation="elu"
    )(x)

    x = layers.Conv2D(
        5,
        (3, 3)
    )(x)

    x = layers.BatchNormalization(
        momentum=0.99,
        epsilon=0.001
    )(x)

    x = layers.ELU()(x)

    x = layers.MaxPooling2D(
        pool_size=(2, 2)
    )(x)

    x = layers.MaxPooling2D(
        pool_size=(5, 5)
    )(x)

    x = layers.Conv2D(
        5,
        (3, 3),
        activation="elu"
    )(x)

    x = layers.Conv2D(
        5,
        (3, 3),
        activation="elu"
    )(x)

    x = layers.Conv2D(
        5,
        (3, 3)
    )(x)

    x = layers.BatchNormalization(
        momentum=0.99,
        epsilon=0.001
    )(x)

    x = layers.ELU()(x)

    x = layers.MaxPooling2D(
        pool_size=(2, 2)
    )(x)

    return inputs, x


# ============================================================
# BUILD R3 MODEL
# ============================================================

def build_model():

    spatial_input, spatial_features = spatial_branch()

    frequency_input, frequency_features = frequency_branch()

    # --------------------------------------------------------
    # Flatten both feature maps
    # --------------------------------------------------------

    spatial_flat = layers.Flatten()(
        spatial_features
    )

    frequency_flat = layers.Flatten()(
        frequency_features
    )

    # --------------------------------------------------------
    # Feature fusion
    # --------------------------------------------------------

    fused = layers.Concatenate(
        name="spatial_frequency_fusion"
    )(
        [spatial_flat, frequency_flat]
    )

    fused = layers.Dropout(
        0.30
    )(fused)

    fused = layers.Dense(
        9
    )(fused)

    fused = layers.ELU()(fused)

    # --------------------------------------------------------
    # Final classifier
    #
    # use_bias=False gives EXACTLY 12,873 parameters
    # --------------------------------------------------------

    outputs = layers.Dense(
        1,
        activation="sigmoid",
        use_bias=False
    )(fused)

    model = Model(
        inputs=[
            spatial_input,
            frequency_input
        ],
        outputs=outputs
    )

    return model


# ============================================================
# SPECIFICITY
# ============================================================

def specificity_score(y_true, y_pred):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    if (tn + fp) == 0:
        return 0.0

    return tn / (tn + fp)


# ============================================================
# MAIN
# ============================================================

if len(sys.argv) != 2:

    print(
        "\nUsage:\n"
        "python r3_spatial_frequency_fusion.py B1\n"
        "python r3_spatial_frequency_fusion.py B2\n"
        "python r3_spatial_frequency_fusion.py B3\n"
        "python r3_spatial_frequency_fusion.py B4\n"
        "python r3_spatial_frequency_fusion.py B5\n"
    )

    sys.exit(1)


benchmark = sys.argv[1].upper()

if benchmark not in BENCHMARKS:

    raise ValueError(
        f"Invalid benchmark: {benchmark}"
    )


print()
print("=" * 70)
print(f"R3 SPATIAL + MULTI-BAND DCT FUSION — {benchmark}")
print("=" * 70)
print()


# ============================================================
# PREPARE DATA
# ============================================================

train_cache, test_cache = prepare_dataset(
    benchmark
)

(
    train_spatial_path,
    train_frequency_path,
    train_labels_path,
    train_shape_spatial,
    train_shape_frequency
) = train_cache

(
    test_spatial_path,
    test_frequency_path,
    test_labels_path,
    test_shape_spatial,
    test_shape_frequency
) = test_cache


train_labels = np.load(
    train_labels_path
)

test_labels = np.load(
    test_labels_path
)

print()
print(f"Train samples: {len(train_labels):,}")
print(f"Test samples : {len(test_labels):,}")
print()


# ============================================================
# GENERATORS
# ============================================================

train_seq = FusionSequence(
    train_spatial_path,
    train_frequency_path,
    train_labels_path,
    train_shape_spatial,
    train_shape_frequency,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_seq = FusionSequence(
    test_spatial_path,
    test_frequency_path,
    test_labels_path,
    test_shape_spatial,
    test_shape_frequency,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# MODEL
# ============================================================

model = build_model()

model.summary()

params = model.count_params()

print()
print("=" * 70)
print(f"PARAMETERS: {params:,}")
print(f"TARGET    : {TARGET_PARAMS:,}")
print("=" * 70)
print()

if params != TARGET_PARAMS:

    raise RuntimeError(
        f"Parameter mismatch: "
        f"{params} != {TARGET_PARAMS}"
    )


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=Nadam(),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# TRAIN
# ============================================================

print()
print("Starting training...")

train_start = time.time()

history = model.fit(
    train_seq,
    epochs=EPOCHS,
    verbose=1
)

train_time = time.time() - train_start


# ============================================================
# INFERENCE
# ============================================================

print()
print("Starting inference...")

inference_start = time.time()

probabilities = model.predict(
    test_seq,
    verbose=1
)

inference_time = time.time() - inference_start


# ============================================================
# PREDICTIONS
# ============================================================

probabilities = probabilities.ravel()

predictions = (
    probabilities >= 0.5
).astype(np.int32)


# ============================================================
# METRICS
# ============================================================

ba = balanced_accuracy_score(
    test_labels,
    predictions
)

precision = precision_score(
    test_labels,
    predictions,
    zero_division=0
)

recall = recall_score(
    test_labels,
    predictions,
    zero_division=0
)

specificity = specificity_score(
    test_labels,
    predictions
)

f1 = f1_score(
    test_labels,
    predictions,
    zero_division=0
)

cm = confusion_matrix(
    test_labels,
    predictions,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print("=" * 70)
print(f"FINAL RESULTS — R3 {benchmark}")
print("=" * 70)

print(
    f"Balanced Accuracy : {ba:.4f} "
    f"({ba * 100:.2f}%)"
)

print(
    f"Precision         : {precision:.4f}"
)

print(
    f"Recall / Sensitivity: {recall:.4f}"
)

print(
    f"Specificity       : {specificity:.4f}"
)

print(
    f"F1 Score          : {f1:.4f}"
)

print()
print("Confusion Matrix:")
print("                Pred NHS   Pred HS")
print(
    f"Actual NHS      {tn:8d}   {fp:8d}"
)
print(
    f"Actual HS       {fn:8d}   {tp:8d}"
)

print()
print(
    f"Training time    : {train_time:.2f} sec"
)

print(
    f"Inference time   : {inference_time:.2f} sec"
)

print(
    f"Parameters       : {params:,}"
)

print("=" * 70)


# ============================================================
# SAVE RESULTS
# ============================================================

results_path = os.path.join(
    CACHE_ROOT,
    f"{benchmark}_results.txt"
)

with open(
    results_path,
    "w"
) as f:

    f.write(
        f"R3 Spatial + Multi-Band DCT Fusion — {benchmark}\n"
    )

    f.write("=" * 70 + "\n")

    f.write(
        f"Balanced Accuracy : {ba:.6f}\n"
    )

    f.write(
        f"Precision         : {precision:.6f}\n"
    )

    f.write(
        f"Recall            : {recall:.6f}\n"
    )

    f.write(
        f"Specificity       : {specificity:.6f}\n"
    )

    f.write(
        f"F1 Score          : {f1:.6f}\n"
    )

    f.write("\nConfusion Matrix:\n")

    f.write(
        f"TN={tn}, FP={fp}, FN={fn}, TP={tp}\n"
    )

    f.write(
        f"\nTraining time : {train_time:.2f} sec\n"
    )

    f.write(
        f"Inference time: {inference_time:.2f} sec\n"
    )

    f.write(
        f"Parameters    : {params}\n"
    )

    f.write(
        f"\nDCT bands:\n"
        f"LOW  <= {LOW_CUTOFF}\n"
        f"MID  <= {MID_CUTOFF}\n"
        f"HIGH > {MID_CUTOFF}\n"
    )

print()
print(f"Results saved to:")
print(results_path)
print()
