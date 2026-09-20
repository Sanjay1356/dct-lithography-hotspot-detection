import os
import sys
import json
import time
import numpy as np
import tensorflow as tf

from PIL import Image
from scipy.fft import dct

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# CONFIG
# ============================================================

VALID_BENCHMARKS = ["B1", "B2", "B3", "B4", "B5"]

if len(sys.argv) != 2:
    print("Usage: python dct_v2.py B1")
    sys.exit(1)

BENCHMARK = sys.argv[1].upper()

if BENCHMARK not in VALID_BENCHMARKS:
    print("Benchmark must be B1, B2, B3, B4, or B5.")
    sys.exit(1)

BENCH_NUM = BENCHMARK[1]

ROOT = os.path.expanduser(
    "~/Downloads/iccad-official"
)

DATASET = os.path.join(
    ROOT,
    f"iccad{BENCH_NUM}"
)

TRAIN_DIR = os.path.join(
    DATASET,
    "train"
)

TEST_DIR = os.path.join(
    DATASET,
    "test"
)

OUTPUT_DIR = os.path.expanduser(
    f"~/Downloads/litho_DCTv2_{BENCHMARK}"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
VAL_SPLIT = 0.20
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# HEADER
# ============================================================

print("\n" + "=" * 65)
print(f"       R1-v2 SIGNED 2-D DCT — {BENCHMARK}")
print("=" * 65)

print("Dataset :", DATASET)
print("Output  :", OUTPUT_DIR)
print("Image   :", IMG_SIZE)
print("Batch   :", BATCH_SIZE)
print("Epochs  :", EPOCHS)
print("Seed    :", SEED)

print("=" * 65)


# ============================================================
# FILE COLLECTION
# ============================================================

def png_files(directory):

    return sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if (
            os.path.isfile(os.path.join(directory, f))
            and f.lower().endswith(".png")
        )
    ])


def collect_train():

    hs = png_files(
        os.path.join(
            TRAIN_DIR,
            "train_hs"
        )
    )

    nhs = png_files(
        os.path.join(
            TRAIN_DIR,
            "train_nhs"
        )
    )

    paths = hs + nhs

    labels = (
        [1] * len(hs) +
        [0] * len(nhs)
    )

    return (
        np.array(paths),
        np.array(labels, dtype=np.int32)
    )


def collect_test():

    hs = png_files(
        os.path.join(
            TEST_DIR,
            "test_hs"
        )
    )

    nhs = png_files(
        os.path.join(
            TEST_DIR,
            "test_nhs"
        )
    )

    paths = hs + nhs

    labels = (
        [1] * len(hs) +
        [0] * len(nhs)
    )

    return (
        np.array(paths),
        np.array(labels, dtype=np.int32)
    )


# ============================================================
# DATA SPLIT
# ============================================================

all_paths, all_labels = collect_train()

train_paths, val_paths, train_labels, val_labels = (
    train_test_split(
        all_paths,
        all_labels,
        test_size=VAL_SPLIT,
        random_state=SEED,
        stratify=all_labels
    )
)

test_paths, test_labels = collect_test()

print("\nTraining:")
print("Total:", len(all_paths))
print("HS:", np.sum(all_labels == 1))
print("NHS:", np.sum(all_labels == 0))

print("\nTrain split:")
print("Total:", len(train_paths))
print("HS:", np.sum(train_labels == 1))
print("NHS:", np.sum(train_labels == 0))

print("\nValidation:")
print("Total:", len(val_paths))
print("HS:", np.sum(val_labels == 1))
print("NHS:", np.sum(val_labels == 0))

print("\nTest:")
print("Total:", len(test_paths))
print("HS:", np.sum(test_labels == 1))
print("NHS:", np.sum(test_labels == 0))


# ============================================================
# TRUE 2-D DCT
# ============================================================

def compute_dct(path):

    image = Image.open(path).convert("L")

    image = image.resize(
        IMG_SIZE,
        Image.Resampling.BILINEAR
    )

    image = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    # TRUE 2-D DCT.
    #
    # First transform rows,
    # then columns.
    #
    # No channel dimension exists here.

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

    return coeff.astype(
        np.float32
    )


# ============================================================
# SIGNED LOG TRANSFORM
# ============================================================

def signed_log_dct(coeff):

    return (
        np.sign(coeff) *
        np.log1p(np.abs(coeff))
    ).astype(
        np.float32
    )


# ============================================================
# TRAINING-SET GLOBAL SCALE
# ============================================================

print("\nComputing training-set DCT scaling...")

# We use only TRAINING images here.
# Validation and test images are never used
# to determine the normalization.

global_abs_max = 0.0

for i, path in enumerate(train_paths):

    coeff = compute_dct(path)

    transformed = signed_log_dct(
        coeff
    )

    local_max = np.max(
        np.abs(transformed)
    )

    if local_max > global_abs_max:
        global_abs_max = float(
            local_max
        )

    if (i + 1) % 100 == 0:
        print(
            f"Scaling scan: "
            f"{i + 1}/{len(train_paths)}"
        )

print(
    "\nGlobal signed-log DCT max:",
    global_abs_max
)

if global_abs_max == 0:
    raise RuntimeError(
        "DCT scaling factor is zero."
    )


# ============================================================
# TENSORFLOW IMAGE LOADER
# ============================================================

def load_dct_image(
    path,
    label
):

    def process(path_bytes):

        path_string = (
            path_bytes.decode("utf-8")
        )

        coeff = compute_dct(
            path_string
        )

        transformed = signed_log_dct(
            coeff
        )

        # Fixed global scaling derived ONLY
        # from the training set.
        transformed = (
            transformed /
            global_abs_max
        )

        # Map approximately [-1, +1]
        # into [0, 1].
        transformed = (
            transformed + 1.0
        ) / 2.0

        transformed = np.clip(
            transformed,
            0.0,
            1.0
        )

        # Replicate to 3 channels so the
        # CNN architecture remains identical.
        transformed = np.repeat(
            transformed[:, :, np.newaxis],
            3,
            axis=2
        )

        return transformed.astype(
            np.float32
        )

    image = tf.numpy_function(
        process,
        [path],
        tf.float32
    )

    image.set_shape(
        [224, 224, 3]
    )

    return (
        image,
        tf.cast(
            label,
            tf.float32
        )
    )


def make_dataset(
    paths,
    labels,
    shuffle=False
):

    ds = tf.data.Dataset.from_tensor_slices(
        (
            paths,
            labels
        )
    )

    if shuffle:

        ds = ds.shuffle(
            buffer_size=len(paths),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    ds = ds.map(
        load_dct_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    ds = ds.batch(
        BATCH_SIZE
    )

    ds = ds.prefetch(
        tf.data.AUTOTUNE
    )

    return ds


train_ds = make_dataset(
    train_paths,
    train_labels,
    shuffle=True
)

val_ds = make_dataset(
    val_paths,
    val_labels,
    shuffle=False
)

test_ds = make_dataset(
    test_paths,
    test_labels,
    shuffle=False
)


# ============================================================
# SAME R0 CNN
# ============================================================

def build_model():

    model = tf.keras.Sequential([

        tf.keras.Input(
            shape=(224, 224, 3)
        ),

        # BLOCK 1

        tf.keras.layers.Conv2D(
            12,
            (3, 3),
            activation="elu"
        ),

        tf.keras.layers.Conv2D(
            12,
            (3, 3),
            activation="elu"
        ),

        tf.keras.layers.Conv2D(
            12,
            (3, 3)
        ),

        tf.keras.layers.BatchNormalization(
            momentum=0.99,
            epsilon=0.001
        ),

        tf.keras.layers.ELU(),

        tf.keras.layers.MaxPooling2D(
            (2, 2)
        ),

        # INTER-BLOCK POOLING

        tf.keras.layers.MaxPooling2D(
            (5, 5)
        ),

        # BLOCK 2

        tf.keras.layers.Conv2D(
            12,
            (3, 3),
            activation="elu"
        ),

        tf.keras.layers.Conv2D(
            12,
            (3, 3),
            activation="elu"
        ),

        tf.keras.layers.Conv2D(
            12,
            (3, 3)
        ),

        tf.keras.layers.BatchNormalization(
            momentum=0.99,
            epsilon=0.001
        ),

        tf.keras.layers.ELU(),

        tf.keras.layers.MaxPooling2D(
            (2, 2)
        ),

        # CLASSIFIER

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dropout(
            0.30
        ),

        tf.keras.layers.Dense(
            10,
            activation="relu"
        ),

        tf.keras.layers.Dense(
            1,
            activation="sigmoid"
        )
    ])

    return model


model = build_model()

model.compile(
    optimizer=tf.keras.optimizers.Nadam(),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# VERIFY
# ============================================================

print("\n" + "=" * 65)
print("MODEL VERIFICATION")
print("=" * 65)

print(
    "Parameters:",
    model.count_params()
)

if model.count_params() != 12873:

    print(
        "ERROR: Expected 12,873 parameters."
    )

    sys.exit(1)

print(
    "Architecture verification PASSED."
)


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 65)
print(f"TRAINING R1-v2 — {BENCHMARK}")
print("=" * 65)

train_start = time.perf_counter()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

training_time = (
    time.perf_counter() -
    train_start
)


# ============================================================
# SAVE MODEL
# ============================================================

model.save(
    os.path.join(
        OUTPUT_DIR,
        f"R1v2_DCT_{BENCHMARK}.keras"
    )
)


with open(
    os.path.join(
        OUTPUT_DIR,
        f"history_{BENCHMARK}.json"
    ),
    "w"
) as f:

    json.dump(
        history.history,
        f,
        indent=4
    )


# ============================================================
# TEST
# ============================================================

print("\n" + "=" * 65)
print(f"TESTING R1-v2 — {BENCHMARK}")
print("=" * 65)

inference_start = time.perf_counter()

probabilities = model.predict(
    test_ds,
    verbose=1
).flatten()

inference_time = (
    time.perf_counter() -
    inference_start
)


# ============================================================
# METRICS
# ============================================================

y_true = test_labels.astype(
    np.int32
)

y_pred = (
    probabilities >= 0.5
).astype(
    np.int32
)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

balanced_accuracy = (
    balanced_accuracy_score(
        y_true,
        y_pred
    )
)

precision = (
    precision_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0
    )
)

recall = (
    recall_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0
    )
)

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0.0
)

f1 = (
    f1_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

results = {

    "benchmark": BENCHMARK,

    "model":
        "R1v2_Signed_2D_DCT",

    "representation":
        "signed_log_2D_DCT",

    "normalization":
        "training_set_global_abs_max",

    "image_size":
        "224x224",

    "batch_size":
        BATCH_SIZE,

    "epochs":
        EPOCHS,

    "seed":
        SEED,

    "validation_split":
        VAL_SPLIT,

    "balanced_accuracy":
        float(balanced_accuracy),

    "HS_precision":
        float(precision),

    "HS_recall_sensitivity":
        float(recall),

    "NHS_specificity":
        float(specificity),

    "HS_F1":
        float(f1),

    "TN": int(tn),
    "FP": int(fp),
    "FN": int(fn),
    "TP": int(tp),

    "parameters":
        int(model.count_params()),

    "training_time_seconds":
        float(training_time),

    "inference_time_seconds":
        float(inference_time),

    "test_samples":
        int(len(y_true)),

    "classification_threshold":
        0.5
}


with open(
    os.path.join(
        OUTPUT_DIR,
        f"results_{BENCHMARK}.json"
    ),
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 65)
print(f"       R1-v2 DCT — {BENCHMARK} RESULTS")
print("=" * 65)

print(
    f"Balanced Accuracy : "
    f"{balanced_accuracy:.4f}"
)

print(
    f"HS Precision      : "
    f"{precision:.4f}"
)

print(
    f"HS Recall         : "
    f"{recall:.4f}"
)

print(
    f"NHS Specificity   : "
    f"{specificity:.4f}"
)

print(
    f"HS F1 Score       : "
    f"{f1:.4f}"
)

print("\nConfusion Matrix")
print("(rows = actual, columns = predicted)")
print()
print("                 NHS       HS")

print(
    f"Actual NHS     {tn:6d}   {fp:6d}"
)

print(
    f"Actual HS      {fn:6d}   {tp:6d}"
)

print(
    f"\nParameters      : "
    f"{model.count_params():,}"
)

print(
    f"Training time   : "
    f"{training_time:.2f} s"
)

print(
    f"Inference time  : "
    f"{inference_time:.2f} s"
)

print(
    f"Test samples    : "
    f"{len(y_true)}"
)

print("\nResults saved to:")
print(OUTPUT_DIR)

print("\n" + "=" * 65)
print("DONE")
print("=" * 65)
