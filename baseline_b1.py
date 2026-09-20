import os
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
# CONFIG
# ============================================================

DATASET = os.path.expanduser(
    "~/Downloads/iccad-official/iccad1"
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
VAL_SPLIT = 0.20
SEED = 42

TRAIN_DIR = os.path.join(DATASET, "train")
TEST_DIR = os.path.join(DATASET, "test")

OUTPUT_DIR = os.path.expanduser(
    "~/Downloads/litho_baseline_B1"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# COLLECT FILES
# ============================================================

def collect_files(directory):

    hs_dir = os.path.join(directory, "train_hs")
    nhs_dir = os.path.join(directory, "train_nhs")

    hs_files = [
        os.path.join(hs_dir, f)
        for f in os.listdir(hs_dir)
        if f.lower().endswith(".png")
    ]

    nhs_files = [
        os.path.join(nhs_dir, f)
        for f in os.listdir(nhs_dir)
        if f.lower().endswith(".png")
    ]

    paths = hs_files + nhs_files

    # IMPORTANT:
    # HS = 1
    # NHS = 0
    labels = [1] * len(hs_files) + [0] * len(nhs_files)

    return np.array(paths), np.array(labels)


def collect_test_files(directory):

    hs_dir = os.path.join(directory, "test_hs")
    nhs_dir = os.path.join(directory, "test_nhs")

    hs_files = [
        os.path.join(hs_dir, f)
        for f in os.listdir(hs_dir)
        if f.lower().endswith(".png")
    ]

    nhs_files = [
        os.path.join(nhs_dir, f)
        for f in os.listdir(nhs_dir)
        if f.lower().endswith(".png")
    ]

    paths = hs_files + nhs_files
    labels = [1] * len(hs_files) + [0] * len(nhs_files)

    return np.array(paths), np.array(labels)


# ============================================================
# LOAD FILE LIST
# ============================================================

train_paths, train_labels = collect_files(TRAIN_DIR)

print("\n================ DATASET B1 ================\n")

print("Total training images:", len(train_paths))
print("Hotspots (HS):", np.sum(train_labels == 1))
print("Non-hotspots (NHS):", np.sum(train_labels == 0))

# Stratified validation split
train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_paths,
    train_labels,
    test_size=VAL_SPLIT,
    random_state=SEED,
    stratify=train_labels
)

print("\nAfter 80/20 split:")

print(
    "Training:",
    len(train_paths),
    "| HS:",
    np.sum(train_labels == 1),
    "| NHS:",
    np.sum(train_labels == 0)
)

print(
    "Validation:",
    len(val_paths),
    "| HS:",
    np.sum(val_labels == 1),
    "| NHS:",
    np.sum(val_labels == 0)
)


# ============================================================
# IMAGE PIPELINE
# ============================================================

def load_image(path, label):

    image = tf.io.read_file(path)

    # Original images are grayscale, but the reference uses
    # RGB input. Decoding with 3 channels replicates the
    # grayscale channel across RGB channels.
    image = tf.image.decode_png(
        image,
        channels=3
    )

    # Match reference target size.
    image = tf.image.resize(
        image,
        IMG_SIZE,
        method="bilinear"
    )

    image = tf.cast(image, tf.float32) / 255.0

    return image, tf.cast(label, tf.float32)


def make_dataset(paths, labels, shuffle=False):

    ds = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if shuffle:
        ds = ds.shuffle(
            buffer_size=len(paths),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    ds = ds.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    ds = ds.batch(BATCH_SIZE)

    ds = ds.prefetch(tf.data.AUTOTUNE)

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

test_paths, test_labels = collect_test_files(TEST_DIR)

test_ds = make_dataset(
    test_paths,
    test_labels,
    shuffle=False
)

print("\nTest images:", len(test_paths))
print("Test HS:", np.sum(test_labels == 1))
print("Test NHS:", np.sum(test_labels == 0))


# ============================================================
# EXACT REFERENCE CNN
# ============================================================

def build_baseline():

    model = tf.keras.Sequential()

    # -------------------------
    # 1st basic block
    # -------------------------

    model.add(
        tf.keras.layers.Conv2D(
            12,
            (3, 3),
            activation="elu",
            input_shape=(224, 224, 3)
        )
    )

    model.add(
        tf.keras.layers.Conv2D(
            12,
            (3, 3),
            activation="elu"
        )
    )

    model.add(
        tf.keras.layers.Conv2D(
            12,
            (3, 3),
            activation=None
        )
    )

    model.add(
        tf.keras.layers.BatchNormalization(
            momentum=0.99,
            epsilon=0.001
        )
    )

    model.add(
        tf.keras.layers.Activation("elu")
    )

    model.add(
        tf.keras.layers.MaxPooling2D(
            (2, 2)
        )
    )

    # -------------------------
    # INTER-BLOCK POOLING
    # -------------------------

    model.add(
        tf.keras.layers.MaxPooling2D(
            (5, 5)
        )
    )

    # -------------------------
    # 2nd basic block
    # -------------------------

    model.add(
        tf.keras.layers.Conv2D(
            12,
            (3, 3),
            activation="elu"
        )
    )

    model.add(
        tf.keras.layers.Conv2D(
            12,
            (3, 3),
            activation="elu"
        )
    )

    model.add(
        tf.keras.layers.Conv2D(
            12,
            (3, 3),
            activation=None
        )
    )

    model.add(
        tf.keras.layers.BatchNormalization(
            momentum=0.99,
            epsilon=0.001
        )
    )

    model.add(
        tf.keras.layers.Activation("elu")
    )

    model.add(
        tf.keras.layers.MaxPooling2D(
            (2, 2)
        )
    )

    # -------------------------
    # Classifier
    # -------------------------

    model.add(
        tf.keras.layers.Flatten()
    )

    model.add(
        tf.keras.layers.Dropout(0.3)
    )

    model.add(
        tf.keras.layers.Dense(
            10,
            activation="relu"
        )
    )

    model.add(
        tf.keras.layers.Dense(
            1,
            activation="sigmoid"
        )
    )

    return model


model = build_baseline()

model.compile(
    optimizer=tf.keras.optimizers.Nadam(),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\n================ MODEL ================\n")

model.summary()


# ============================================================
# VERIFY PARAMETER COUNT
# ============================================================

print(
    "\nPARAMETER COUNT:",
    model.count_params()
)

assert model.count_params() == 12873, (
    f"Unexpected parameter count: "
    f"{model.count_params()}"
)


# ============================================================
# TRAIN
# ============================================================

print("\n================ TRAINING B1 ================\n")

start_train = time.perf_counter()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

training_time = (
    time.perf_counter() - start_train
)


# ============================================================
# SAVE MODEL
# ============================================================

model.save(
    os.path.join(
        OUTPUT_DIR,
        "baseline_B1_exact.keras"
    )
)

with open(
    os.path.join(
        OUTPUT_DIR,
        "training_history_exact.json"
    ),
    "w"
) as f:

    json.dump(
        history.history,
        f
    )


# ============================================================
# TEST
# ============================================================

print("\n================ TESTING B1 ================\n")

start_inference = time.perf_counter()

y_prob = model.predict(
    test_ds,
    verbose=1
).flatten()

inference_time = (
    time.perf_counter() - start_inference
)

y_true = test_labels.astype(int)

# HS = positive class
y_pred = (
    y_prob >= 0.5
).astype(int)


# ============================================================
# METRICS
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

balanced_acc = balanced_accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    pos_label=1,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    pos_label=1,
    zero_division=0
)

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0
)

f1 = f1_score(
    y_true,
    y_pred,
    pos_label=1,
    zero_division=0
)


# ============================================================
# RESULTS
# ============================================================

metrics = {

    "benchmark": "B1",
    "model": "R0_spatial_reference_CNN",

    "balanced_accuracy":
        float(balanced_acc),

    "precision_HS":
        float(precision),

    "recall_sensitivity_HS":
        float(recall),

    "specificity_NHS":
        float(specificity),

    "f1_HS":
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

    "threshold":
        0.5,

    "seed":
        SEED
}

with open(
    os.path.join(
        OUTPUT_DIR,
        "metrics_exact.json"
    ),
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


# ============================================================
# PRINT
# ============================================================

print("\n============================================")
print("       R0 BASELINE — B1 EXACT RESULTS")
print("============================================")

print(
    f"Balanced Accuracy : {balanced_acc:.4f}"
)

print(
    f"HS Precision      : {precision:.4f}"
)

print(
    f"HS Recall         : {recall:.4f}"
)

print(
    f"NHS Specificity   : {specificity:.4f}"
)

print(
    f"HS F1 Score       : {f1:.4f}"
)

print("\nConfusion Matrix")
print("(rows = actual, columns = predicted)")
print("              NHS     HS")
print(
    f"Actual NHS    {tn:5d}  {fp:5d}"
)
print(
    f"Actual HS     {fn:5d}  {tp:5d}"
)

print(
    f"\nParameters    : {model.count_params():,}"
)

print(
    f"Training time : {training_time:.2f} s"
)

print(
    f"Inference time: {inference_time:.2f} s"
)

print(
    f"Test samples  : {len(y_true)}"
)

print("\nResults saved to:")
print(OUTPUT_DIR)

print("\nDONE.")
