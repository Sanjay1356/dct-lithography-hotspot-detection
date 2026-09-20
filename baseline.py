import os
import sys
import json
import time
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# CONFIGURATION
# ============================================================

VALID_BENCHMARKS = ["B1", "B2", "B3", "B4", "B5"]

if len(sys.argv) != 2:
    print("\nUsage:")
    print("    python baseline.py B1")
    print("    python baseline.py B2")
    print("    python baseline.py B3")
    print("    python baseline.py B4")
    print("    python baseline.py B5\n")
    sys.exit(1)

BENCHMARK = sys.argv[1].upper()

if BENCHMARK not in VALID_BENCHMARKS:
    print(f"\nERROR: Invalid benchmark '{BENCHMARK}'")
    print("Choose from: B1, B2, B3, B4, B5\n")
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
    f"~/Downloads/litho_baseline_{BENCHMARK}"
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


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# PRINT CONFIGURATION
# ============================================================

print("\n")
print("=" * 60)
print(f"        R0 SPATIAL CNN — {BENCHMARK}")
print("=" * 60)

print(f"Dataset : {DATASET_DIR}")
print(f"Output  : {OUTPUT_DIR}")
print(f"Image   : {IMG_SIZE}")
print(f"Batch   : {BATCH_SIZE}")
print(f"Epochs  : {EPOCHS}")
print(f"Seed    : {SEED}")

print("=" * 60)


# ============================================================
# FILE COLLECTION
# ============================================================

def get_png_files(directory):

    if not os.path.isdir(directory):
        raise FileNotFoundError(
            f"Directory does not exist:\n{directory}"
        )

    files = []

    for filename in os.listdir(directory):

        full_path = os.path.join(
            directory,
            filename
        )

        if (
            os.path.isfile(full_path)
            and filename.lower().endswith(".png")
        ):
            files.append(full_path)

    return sorted(files)


def collect_training_data():

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

    # HS = 1
    # NHS = 0

    paths = (
        hs_files +
        nhs_files
    )

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

    paths = (
        hs_files +
        nhs_files
    )

    labels = (
        [1] * len(hs_files) +
        [0] * len(nhs_files)
    )

    return (
        np.array(paths),
        np.array(labels, dtype=np.int32)
    )


# ============================================================
# LOAD DATA
# ============================================================

print("\nCollecting training files...")

all_train_paths, all_train_labels = (
    collect_training_data()
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


# ============================================================
# STRATIFIED TRAIN / VALIDATION SPLIT
# ============================================================

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

print("\nCollecting test files...")

test_paths, test_labels = (
    collect_test_data()
)

print(
    f"Test       : {len(test_paths)} "
    f"(HS={np.sum(test_labels == 1)}, "
    f"NHS={np.sum(test_labels == 0)})"
)


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_png(
        image,
        channels=3
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
        load_image,
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
# REFERENCE LIGHTWEIGHT CNN
# ============================================================

def build_model():

    model = tf.keras.Sequential([

        tf.keras.Input(
            shape=(224, 224, 3)
        ),

        # ----------------------------------------
        # BLOCK 1
        # ----------------------------------------

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
            pool_size=(2, 2)
        ),

        # ----------------------------------------
        # 5x5 POOLING BETWEEN BLOCKS
        # ----------------------------------------

        tf.keras.layers.MaxPooling2D(
            pool_size=(5, 5)
        ),

        # ----------------------------------------
        # BLOCK 2
        # ----------------------------------------

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
            pool_size=(2, 2)
        ),

        # ----------------------------------------
        # CLASSIFIER
        # ----------------------------------------

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


# ============================================================
# BUILD MODEL
# ============================================================

model = build_model()

model.compile(
    optimizer=tf.keras.optimizers.Nadam(),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# VERIFY ARCHITECTURE
# ============================================================

print("\n")
print("=" * 60)
print("MODEL SUMMARY")
print("=" * 60)

model.summary()

print(
    f"\nPARAMETER COUNT: "
    f"{model.count_params():,}"
)

if model.count_params() != 12873:

    print("\nWARNING!")
    print(
        "Expected 12,873 parameters, "
        f"but got {model.count_params()}."
    )

    print(
        "STOPPING before training."
    )

    sys.exit(1)

print(
    "Architecture verification PASSED."
)


# ============================================================
# TRAIN
# ============================================================

print("\n")
print("=" * 60)
print(f"TRAINING {BENCHMARK}")
print("=" * 60)

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

model_path = os.path.join(
    OUTPUT_DIR,
    f"R0_{BENCHMARK}.keras"
)

model.save(
    model_path
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_path = os.path.join(
    OUTPUT_DIR,
    f"history_{BENCHMARK}.json"
)

with open(
    history_path,
    "w"
) as file:

    json.dump(
        history.history,
        file,
        indent=4
    )


# ============================================================
# TEST
# ============================================================

print("\n")
print("=" * 60)
print(f"TESTING {BENCHMARK}")
print("=" * 60)

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
# PREDICTIONS
# ============================================================

y_true = test_labels.astype(
    np.int32
)

y_pred = (
    probabilities >= 0.5
).astype(np.int32)


# ============================================================
# CONFUSION MATRIX
# ============================================================

# Label convention:
#
# NHS = 0
# HS  = 1
#
# rows    = actual
# columns = predicted

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()


# ============================================================
# METRICS
# ============================================================

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
# RESULTS
# ============================================================

results = {

    "benchmark":
        BENCHMARK,

    "model":
        "R0_Spatial_Reference_CNN",

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


results_path = os.path.join(
    OUTPUT_DIR,
    f"results_{BENCHMARK}.json"
)

with open(
    results_path,
    "w"
) as file:

    json.dump(
        results,
        file,
        indent=4
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 60)
print(f"       R0 BASELINE — {BENCHMARK} RESULTS")
print("=" * 60)

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

print("\nSaved:")
print(model_path)
print(history_path)
print(results_path)

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
