import os
import sys
import json
import time
import numpy as np
import tensorflow as tf

from scipy.fft import dctn

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
    print("\nUsage:")
    print("    python dct_baseline.py B1")
    print("    python dct_baseline.py B2")
    print("    python dct_baseline.py B3")
    print("    python dct_baseline.py B4")
    print("    python dct_baseline.py B5\n")
    sys.exit(1)

BENCHMARK = sys.argv[1].upper()

if BENCHMARK not in VALID_BENCHMARKS:
    print("ERROR: Benchmark must be B1, B2, B3, B4, or B5.")
    sys.exit(1)

BENCH_NUM = BENCHMARK[1]

DATASET_ROOT = os.path.expanduser(
    "~/Downloads/iccad-official"
)

DATASET_DIR = os.path.join(
    DATASET_ROOT,
    f"iccad{BENCH_NUM}"
)

TRAIN_DIR = os.path.join(
    DATASET_DIR,
    "train"
)

TEST_DIR = os.path.join(
    DATASET_DIR,
    "test"
)

OUTPUT_DIR = os.path.expanduser(
    f"~/Downloads/litho_DCT_{BENCHMARK}"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
VALIDATION_SPLIT = 0.20
SEED = 42


np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# HEADER
# ============================================================

print("\n")
print("=" * 65)
print(f"        R1 DCT-ONLY CNN — {BENCHMARK}")
print("=" * 65)
print(f"Dataset : {DATASET_DIR}")
print(f"Output  : {OUTPUT_DIR}")
print(f"Image   : {IMG_SIZE}")
print(f"Batch   : {BATCH_SIZE}")
print(f"Epochs  : {EPOCHS}")
print(f"Seed    : {SEED}")
print("=" * 65)


# ============================================================
# FILE COLLECTION
# ============================================================

def get_png_files(directory):

    if not os.path.isdir(directory):
        raise FileNotFoundError(
            f"Directory does not exist:\n{directory}"
        )

    return sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if (
            os.path.isfile(os.path.join(directory, f))
            and f.lower().endswith(".png")
        )
    ])


def collect_data(base_dir):

    hs_dir = os.path.join(
        base_dir,
        "hs"
    )

    nhs_dir = os.path.join(
        base_dir,
        "nhs"
    )

    hs_files = get_png_files(hs_dir)
    nhs_files = get_png_files(nhs_dir)

    paths = hs_files + nhs_files

    # HS = 1
    # NHS = 0

    labels = (
        [1] * len(hs_files) +
        [0] * len(nhs_files)
    )

    return (
        np.array(paths),
        np.array(labels, dtype=np.int32)
    )


def collect_train_data():

    hs_dir = os.path.join(
        TRAIN_DIR,
        "train_hs"
    )

    nhs_dir = os.path.join(
        TRAIN_DIR,
        "train_nhs"
    )

    hs_files = get_png_files(hs_dir)
    nhs_files = get_png_files(nhs_dir)

    paths = hs_files + nhs_files

    labels = (
        [1] * len(hs_files) +
        [0] * len(nhs_files)
    )

    return (
        np.array(paths),
        np.array(labels, dtype=np.int32)
    )


def collect_test_data():

    hs_dir = os.path.join(
        TEST_DIR,
        "test_hs"
    )

    nhs_dir = os.path.join(
        TEST_DIR,
        "test_nhs"
    )

    hs_files = get_png_files(hs_dir)
    nhs_files = get_png_files(nhs_dir)

    paths = hs_files + nhs_files

    labels = (
        [1] * len(hs_files) +
        [0] * len(nhs_files)
    )

    return (
        np.array(paths),
        np.array(labels, dtype=np.int32)
    )


# ============================================================
# TRAIN / VALIDATION
# ============================================================

print("\nCollecting training files...")

all_train_paths, all_train_labels = (
    collect_train_data()
)

print(
    f"Total training images : "
    f"{len(all_train_paths)}"
)

print(
    f"Training HS            : "
    f"{np.sum(all_train_labels == 1)}"
)

print(
    f"Training NHS           : "
    f"{np.sum(all_train_labels == 0)}"
)


train_paths, val_paths, train_labels, val_labels = (
    train_test_split(
        all_train_paths,
        all_train_labels,
        test_size=VALIDATION_SPLIT,
        random_state=SEED,
        stratify=all_train_labels
    )
)


print("\nTrain / Validation split:")

print(
    f"Train      : {len(train_paths)} "
    f"(HS={np.sum(train_labels == 1)}, "
    f"NHS={np.sum(train_labels == 0)})"
)

print(
    f"Validation : {len(val_paths)} "
    f"(HS={np.sum(val_labels == 1)}, "
    f"NHS={np.sum(val_labels == 0)})"
)


# ============================================================
# TEST DATA
# ============================================================

test_paths, test_labels = (
    collect_test_data()
)

print(
    f"\nTest       : {len(test_paths)} "
    f"(HS={np.sum(test_labels == 1)}, "
    f"NHS={np.sum(test_labels == 0)})"
)


# ============================================================
# DCT PREPROCESSING
# ============================================================

def load_dct_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_png(
        image,
        channels=1
    )

    image = tf.image.resize(
        image,
        IMG_SIZE,
        method="bilinear"
    )

    image = tf.cast(
        image,
        tf.float32
    ) / 255.0

    # Convert TensorFlow tensor to NumPy through
    # tf.numpy_function so scipy DCT can be used.
    def compute_dct(img):

        img = img.astype(
            np.float32
        )

        # 2D orthonormal DCT
        coeff = dctn(
            img,
            type=2,
            norm="ortho"
        )

        # Magnitude / energy representation
        coeff = np.abs(coeff)

        # Compress dynamic range
        coeff = np.log1p(coeff)

        # Per-image normalization
        min_val = coeff.min()
        max_val = coeff.max()

        if max_val > min_val:

            coeff = (
                (coeff - min_val) /
                (max_val - min_val)
            )

        else:

            coeff = np.zeros_like(
                coeff
            )

        return coeff.astype(
            np.float32
        )

    image = tf.numpy_function(
        compute_dct,
        [image],
        tf.float32
    )

    image.set_shape(
        [224, 224, 1]
    )

    # Replicate DCT channel to match the
    # 3-channel CNN input.
    image = tf.repeat(
        image,
        repeats=3,
        axis=-1
    )

    return (
        image,
        tf.cast(label, tf.float32)
    )


def make_dataset(
    paths,
    labels,
    shuffle=False
):

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            paths,
            labels
        )
    )

    if shuffle:

        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_dct_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


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
# SAME LIGHTWEIGHT CNN
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
# VERIFY PARAMETER COUNT
# ============================================================

print("\n")
print("=" * 65)
print("MODEL VERIFICATION")
print("=" * 65)

model.summary()

print(
    f"\nPARAMETER COUNT: "
    f"{model.count_params():,}"
)

if model.count_params() != 12873:

    print(
        "\nERROR: Parameter count is not 12,873."
    )

    print(
        "Stopping before training."
    )

    sys.exit(1)

print(
    "Architecture verification PASSED."
)


# ============================================================
# TRAIN
# ============================================================

print("\n")
print("=" * 65)
print(f"TRAINING R1 — {BENCHMARK}")
print("=" * 65)

start_train = time.perf_counter()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

training_time = (
    time.perf_counter() -
    start_train
)


# ============================================================
# SAVE MODEL
# ============================================================

model.save(
    os.path.join(
        OUTPUT_DIR,
        f"R1_DCT_{BENCHMARK}.keras"
    )
)


# ============================================================
# SAVE HISTORY
# ============================================================

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

print("\n")
print("=" * 65)
print(f"TESTING R1 — {BENCHMARK}")
print("=" * 65)

start_inference = time.perf_counter()

probabilities = model.predict(
    test_ds,
    verbose=1
).flatten()

inference_time = (
    time.perf_counter() -
    start_inference
)


# ============================================================
# METRICS
# ============================================================

y_true = test_labels.astype(
    np.int32
)

y_pred = (
    probabilities >= 0.5
).astype(np.int32)


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

    "benchmark":
        BENCHMARK,

    "model":
        "R1_DCT_only",

    "representation":
        "log_magnitude_2D_DCT",

    "image_size":
        "224x224",

    "batch_size":
        BATCH_SIZE,

    "epochs":
        EPOCHS,

    "seed":
        SEED,

    "validation_split":
        VALIDATION_SPLIT,

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

    "TN":
        int(tn),

    "FP":
        int(fp),

    "FN":
        int(fn),

    "TP":
        int(tp),

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
# FINAL RESULTS
# ============================================================

print("\n")
print("=" * 65)
print(f"       R1 DCT-ONLY — {BENCHMARK} RESULTS")
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

print("\n")
print("=" * 65)
print("DONE")
print("=" * 65)
