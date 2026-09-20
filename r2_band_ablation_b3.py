import os
import sys
import time
import random
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Nadam

from sklearn.metrics import (
    confusion_matrix,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# DCT BAND ABLATION — B3
# ============================================================

BENCHMARK = "B3"

DATA_ROOT = os.path.expanduser(
    "~/Downloads/iccad-official"
)

CACHE_ROOT = os.path.expanduser(
    "~/Downloads/litho_ablation_cache"
)

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42


# ============================================================
# COMMAND-LINE ARGUMENT
# ============================================================

if len(sys.argv) != 2:

    print(
        "Usage: "
        "python r2_band_ablation_b3.py "
        "LOW|MID|HIGH"
    )

    sys.exit(1)


BAND = sys.argv[1].upper()

if BAND not in {"LOW", "MID", "HIGH"}:

    print(
        "ERROR: BAND must be LOW, MID, or HIGH"
    )

    sys.exit(1)


# ============================================================
# REPRODUCIBILITY
# ============================================================

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# DATASET PATHS
# ============================================================

BENCH_DIR = os.path.join(
    DATA_ROOT,
    "iccad3"
)

TRAIN_HS = os.path.join(
    BENCH_DIR,
    "train",
    "train_hs"
)

TRAIN_NHS = os.path.join(
    BENCH_DIR,
    "train",
    "train_nhs"
)

TEST_HS = os.path.join(
    BENCH_DIR,
    "test",
    "test_hs"
)

TEST_NHS = os.path.join(
    BENCH_DIR,
    "test",
    "test_nhs"
)


# ============================================================
# CACHE PATHS
# ============================================================

CACHE_DIR = os.path.join(
    CACHE_ROOT,
    BAND
)

os.makedirs(
    CACHE_DIR,
    exist_ok=True
)

TRAIN_CACHE = os.path.join(
    CACHE_DIR,
    "train.npy"
)

TEST_CACHE = os.path.join(
    CACHE_DIR,
    "test.npy"
)

TRAIN_LABELS = os.path.join(
    CACHE_DIR,
    "train_labels.npy"
)

TEST_LABELS = os.path.join(
    CACHE_DIR,
    "test_labels.npy"
)


# ============================================================
# HEADER
# ============================================================

print("\n" + "=" * 70)
print(
    f"DCT BAND ABLATION — "
    f"{BENCHMARK} — {BAND}"
)
print("=" * 70)

print("\nFrequency bands:")
print("  LOW  : r <= 0.15")
print("  MID  : 0.15 < r <= 0.40")
print("  HIGH : r > 0.40")


# ============================================================
# EXACT R2 BAND MASK
# ============================================================

def make_band_mask():

    y = np.arange(IMG_SIZE)
    x = np.arange(IMG_SIZE)

    yy, xx = np.meshgrid(
        y,
        x,
        indexing="ij"
    )

    # IMPORTANT:
    # R2 used coordinates normalized by
    # (IMG_SIZE - 1), not IMG_SIZE.
    #
    # For 224x224:
    # denominator = 223
    #
    # This reproduces the exact locked R2
    # coefficient counts:
    #
    # LOW  = 1,806
    # MID  = 10,822
    # HIGH = 37,548

    denom = IMG_SIZE - 1

    fy = yy / denom
    fx = xx / denom

    r = (
        np.sqrt(
            fy ** 2 +
            fx ** 2
        )
        /
        np.sqrt(2.0)
    )

    if BAND == "LOW":

        mask = (
            r <= 0.15
        )

    elif BAND == "MID":

        mask = (
            (r > 0.15)
            &
            (r <= 0.40)
        )

    else:

        mask = (
            r > 0.40
        )

    return mask.astype(
        np.float32
    )


BAND_MASK = make_band_mask()

selected_coefficients = int(
    BAND_MASK.sum()
)

total_coefficients = (
    IMG_SIZE * IMG_SIZE
)


print("\nBand coefficients:")
print(
    f"  Selected : "
    f"{selected_coefficients:,}"
)

print(
    f"  Total    : "
    f"{total_coefficients:,}"
)


# ============================================================
# HARD SANITY CHECK
# ============================================================

EXPECTED_COUNTS = {
    "LOW": 1806,
    "MID": 10822,
    "HIGH": 37548,
}

expected = EXPECTED_COUNTS[BAND]

if selected_coefficients != expected:

    raise RuntimeError(
        "\n"
        "FATAL: Band-mask mismatch!\n"
        f"Band: {BAND}\n"
        f"Expected coefficients: "
        f"{expected:,}\n"
        f"Actual coefficients:   "
        f"{selected_coefficients:,}\n\n"
        "Do NOT run the experiment."
    )

print(
    "  ✓ Band definition matches R2"
)


# ============================================================
# GET PNG FILES
# ============================================================

def get_images(folder):

    files = []

    for filename in sorted(
        os.listdir(folder)
    ):

        path = os.path.join(
            folder,
            filename
        )

        if (
            os.path.isfile(path)
            and
            filename.lower().endswith(
                ".png"
            )
        ):

            files.append(path)

    return files


# ============================================================
# COLLECT DATASET
# ============================================================

def collect_dataset():

    train_paths = (
        [
            (path, 1)
            for path in get_images(
                TRAIN_HS
            )
        ]
        +
        [
            (path, 0)
            for path in get_images(
                TRAIN_NHS
            )
        ]
    )

    test_paths = (
        [
            (path, 1)
            for path in get_images(
                TEST_HS
            )
        ]
        +
        [
            (path, 0)
            for path in get_images(
                TEST_NHS
            )
        ]
    )

    return (
        train_paths,
        test_paths
    )


# ============================================================
# 2D ORTHONORMAL DCT
# ============================================================

def dct2d(image):

    # TensorFlow's DCT operates along
    # the last axis only.
    #
    # Therefore perform:
    #
    # 1. DCT along columns
    # 2. Transpose
    # 3. DCT along the new last axis
    # 4. Transpose back

    dct = tf.signal.dct(
        image,
        type=2,
        norm="ortho",
        axis=-1
    )

    dct = tf.transpose(
        dct
    )

    dct = tf.signal.dct(
        dct,
        type=2,
        norm="ortho",
        axis=-1
    )

    dct = tf.transpose(
        dct
    )

    return dct


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def process_image(path):

    # Read grayscale PNG

    image = tf.io.read_file(
        path
    )

    image = tf.image.decode_png(
        image,
        channels=1
    )

    # Resize to 224 × 224

    image = tf.image.resize(
        image,
        [
            IMG_SIZE,
            IMG_SIZE
        ],
        method="bilinear"
    )

    # Normalize to [0, 1]

    image = (
        tf.cast(
            image,
            tf.float32
        )
        /
        255.0
    )

    image = tf.squeeze(
        image,
        axis=-1
    )

    # 2D orthonormal DCT

    dct = dct2d(
        image
    )

    # Magnitude

    dct = tf.abs(
        dct
    )

    # Log compression

    dct = tf.math.log1p(
        dct
    )

    # Per-image min-max normalization

    dmin = tf.reduce_min(
        dct
    )

    dmax = tf.reduce_max(
        dct
    )

    dct = (
        dct - dmin
    ) / (
        dmax - dmin + 1e-8
    )

    # Keep ONLY the selected frequency band

    dct = (
        dct
        *
        tf.constant(
            BAND_MASK,
            dtype=tf.float32
        )
    )

    return dct.numpy().astype(
        np.float16
    )


# ============================================================
# CACHE CREATION
# ============================================================

def create_cache(
    paths,
    cache_path,
    labels_path,
    name
):

    if (
        os.path.exists(cache_path)
        and
        os.path.exists(labels_path)
    ):

        print(
            f"\nUsing existing cache: "
            f"{name}"
        )

        data = np.load(
            cache_path,
            mmap_mode="r"
        )

        labels = np.load(
            labels_path
        )

        return (
            data,
            labels
        )

    print(
        f"\nCreating cache: "
        f"{name}"
    )

    print(
        f"Images: "
        f"{len(paths):,}"
    )

    data = np.lib.format.open_memmap(
        cache_path,
        mode="w+",
        dtype=np.float16,
        shape=(
            len(paths),
            IMG_SIZE,
            IMG_SIZE
        )
    )

    labels = np.empty(
        len(paths),
        dtype=np.int8
    )

    start = time.time()

    for i, (
        path,
        label
    ) in enumerate(paths):

        data[i] = process_image(
            path
        )

        labels[i] = label

        if (
            (i + 1) % 500 == 0
            or
            i == len(paths) - 1
        ):

            elapsed = (
                time.time()
                -
                start
            )

            print(
                f"Processed "
                f"{i + 1:,}/"
                f"{len(paths):,} "
                f"("
                f"{100 * (i + 1) / len(paths):.1f}"
                f"%) "
                f"[{elapsed:.1f}s]"
            )

    data.flush()

    np.save(
        labels_path,
        labels
    )

    print(
        f"Cache written: "
        f"{name}"
    )

    return (
        data,
        labels
    )


# ============================================================
# DATA GENERATOR
# ============================================================

class DCTSequence(
    tf.keras.utils.Sequence
):

    def __init__(
        self,
        data,
        labels,
        batch_size=32,
        shuffle=False,
        **kwargs
    ):

        super().__init__(
            **kwargs
        )

        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.indices = np.arange(
            len(labels)
        )

        if self.shuffle:

            np.random.shuffle(
                self.indices
            )

    def __len__(self):

        return int(
            np.ceil(
                len(self.labels)
                /
                self.batch_size
            )
        )

    def __getitem__(
        self,
        index
    ):

        batch_indices = (
            self.indices[
                index * self.batch_size:
                min(
                    (index + 1)
                    * self.batch_size,
                    len(self.labels)
                )
            ]
        )

        x = self.data[
            batch_indices
        ]

        # Single channel → 3 identical channels

        x = (
            x[..., np.newaxis]
        )

        x = np.repeat(
            x,
            3,
            axis=-1
        )

        y = self.labels[
            batch_indices
        ].astype(
            np.float32
        )

        return (
            x.astype(
                np.float32
            ),
            y
        )

    def on_epoch_end(self):

        if self.shuffle:

            np.random.shuffle(
                self.indices
            )


# ============================================================
# EXACT 12,873-PARAMETER CNN
# ============================================================

def build_model():

    inputs = layers.Input(
        shape=(
            IMG_SIZE,
            IMG_SIZE,
            3
        )
    )

    # ========================================================
    # BLOCK 1
    # ========================================================

    x = layers.Conv2D(
        12,
        (3, 3),
        padding="valid",
        activation="elu"
    )(inputs)

    x = layers.Conv2D(
        12,
        (3, 3),
        padding="valid",
        activation="elu"
    )(x)

    x = layers.Conv2D(
        12,
        (3, 3),
        padding="valid",
        activation=None
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

    # ========================================================
    # BLOCK 2
    # ========================================================

    x = layers.Conv2D(
        12,
        (3, 3),
        padding="valid",
        activation="elu"
    )(x)

    x = layers.Conv2D(
        12,
        (3, 3),
        padding="valid",
        activation="elu"
    )(x)

    x = layers.Conv2D(
        12,
        (3, 3),
        padding="valid",
        activation=None
    )(x)

    x = layers.BatchNormalization(
        momentum=0.99,
        epsilon=0.001
    )(x)

    x = layers.ELU()(x)

    x = layers.MaxPooling2D(
        pool_size=(2, 2)
    )(x)

    # ========================================================
    # CLASSIFIER
    # ========================================================

    x = layers.Flatten()(x)

    x = layers.Dropout(
        0.3
    )(x)

    x = layers.Dense(
        10
    )(x)

    outputs = layers.Dense(
        1,
        activation="sigmoid"
    )(x)

    return Model(
        inputs,
        outputs
    )


# ============================================================
# MAIN
# ============================================================

train_paths, test_paths = (
    collect_dataset()
)

print(
    f"\nTrain samples: "
    f"{len(train_paths):,}"
)

print(
    f"Test samples : "
    f"{len(test_paths):,}"
)


# ============================================================
# CACHE
# ============================================================

train_data, train_labels = (
    create_cache(
        train_paths,
        TRAIN_CACHE,
        TRAIN_LABELS,
        "B3_train"
    )
)

test_data, test_labels = (
    create_cache(
        test_paths,
        TEST_CACHE,
        TEST_LABELS,
        "B3_test"
    )
)


# ============================================================
# MODEL
# ============================================================

model = build_model()

print(
    "\n"
    +
    "=" * 70
)

print("MODEL")

print(
    "=" * 70
)

model.summary()

params = model.count_params()

print(
    "\n"
    +
    "=" * 70
)

print(
    f"PARAMETERS: "
    f"{params:,}"
)

print(
    "TARGET    : "
    "12,873"
)

print(
    "=" * 70
)

if params != 12873:

    raise RuntimeError(
        f"Parameter mismatch: "
        f"{params:,} != 12,873"
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
# DATA GENERATORS
# ============================================================

train_seq = DCTSequence(
    train_data,
    train_labels,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_seq = DCTSequence(
    test_data,
    test_labels,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# TRAINING
# ============================================================

print(
    "\nStarting training..."
)

train_start = time.time()

model.fit(
    train_seq,
    epochs=EPOCHS,
    verbose=1
)

train_time = (
    time.time()
    -
    train_start
)


# ============================================================
# INFERENCE
# ============================================================

print(
    "\nStarting inference..."
)

inference_start = time.time()

pred_prob = model.predict(
    test_seq,
    verbose=1
).ravel()

inference_time = (
    time.time()
    -
    inference_start
)

pred = (
    pred_prob >= 0.5
).astype(int)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    test_labels,
    pred,
    labels=[0, 1]
)

TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]
TP = cm[1, 1]


# ============================================================
# METRICS
# ============================================================

balanced_accuracy = (
    balanced_accuracy_score(
        test_labels,
        pred
    )
)

precision = precision_score(
    test_labels,
    pred,
    zero_division=0
)

recall = recall_score(
    test_labels,
    pred,
    zero_division=0
)

specificity = (
    TN /
    (TN + FP)
    if (TN + FP) > 0
    else 0.0
)

f1 = f1_score(
    test_labels,
    pred,
    zero_division=0
)


# ============================================================
# FINAL RESULTS
# ============================================================

print(
    "\n"
    +
    "=" * 70
)

print(
    f"FINAL RESULTS — "
    f"B3 {BAND} ONLY"
)

print(
    "=" * 70
)

print(
    f"Balanced Accuracy : "
    f"{balanced_accuracy:.4f} "
    f"({balanced_accuracy * 100:.2f}%)"
)

print(
    f"Precision         : "
    f"{precision:.4f}"
)

print(
    f"Recall / Sensitivity: "
    f"{recall:.4f}"
)

print(
    f"Specificity       : "
    f"{specificity:.4f}"
)

print(
    f"F1 Score          : "
    f"{f1:.4f}"
)

print(
    "\nConfusion Matrix:"
)

print(
    "                "
    "Pred NHS   Pred HS"
)

print(
    f"Actual NHS      "
    f"{TN:8d}   "
    f"{FP:8d}"
)

print(
    f"Actual HS       "
    f"{FN:8d}   "
    f"{TP:8d}"
)

print(
    f"\nTraining time    : "
    f"{train_time:.2f} sec"
)

print(
    f"Inference time   : "
    f"{inference_time:.2f} sec"
)

print(
    f"Parameters       : "
    f"{params:,}"
)

print(
    "=" * 70
)


# ============================================================
# SAVE RESULTS
# ============================================================

result_file = os.path.join(
    CACHE_DIR,
    "results.txt"
)

with open(
    result_file,
    "w"
) as f:

    f.write(
        f"B3 {BAND} ONLY ABLATION\n"
    )

    f.write(
        "=" * 60 + "\n"
    )

    f.write(
        f"Selected coefficients: "
        f"{selected_coefficients}\n"
    )

    f.write(
        f"Balanced Accuracy : "
        f"{balanced_accuracy:.4f} "
        f"({balanced_accuracy * 100:.2f}%)\n"
    )

    f.write(
        f"Precision         : "
        f"{precision:.4f}\n"
    )

    f.write(
        f"Recall            : "
        f"{recall:.4f}\n"
    )

    f.write(
        f"Specificity       : "
        f"{specificity:.4f}\n"
    )

    f.write(
        f"F1                : "
        f"{f1:.4f}\n\n"
    )

    f.write(
        "Confusion Matrix:\n"
    )

    f.write(
        f"TN={TN}, "
        f"FP={FP}, "
        f"FN={FN}, "
        f"TP={TP}\n\n"
    )

    f.write(
        f"Training time     : "
        f"{train_time:.2f}s\n"
    )

    f.write(
        f"Inference time    : "
        f"{inference_time:.2f}s\n"
    )

    f.write(
        f"Parameters        : "
        f"{params}\n"
    )

print(
    f"\nResults saved to:\n"
    f"{result_file}"
)
