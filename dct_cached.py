import os
import sys
import time
import numpy as np
import tensorflow as tf

from scipy.fft import dctn
from tensorflow.keras import layers, models
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = os.path.expanduser(
    "~/Downloads/iccad-official"
)

CACHE_ROOT = os.path.expanduser(
    "~/Downloads/litho_dct_cache"
)

BENCHMARK_MAP = {
    "B1": "iccad1",
    "B2": "iccad2",
    "B3": "iccad3",
    "B4": "iccad4",
    "B5": "iccad5"
}

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# BENCHMARK
# ============================================================

if len(sys.argv) != 2:

    print()
    print("Usage:")
    print("python dct_cached.py B1")
    print("python dct_cached.py B2")
    print("python dct_cached.py B3")
    print("python dct_cached.py B4")
    print("python dct_cached.py B5")
    print()

    sys.exit(1)

benchmark = sys.argv[1].upper()

if benchmark not in BENCHMARK_MAP:

    print(f"Invalid benchmark: {benchmark}")
    sys.exit(1)

dataset_name = BENCHMARK_MAP[benchmark]

DATASET_PATH = os.path.join(
    DATASET_ROOT,
    dataset_name
)

TRAIN_PATH = os.path.join(
    DATASET_PATH,
    "train"
)

TEST_PATH = os.path.join(
    DATASET_PATH,
    "test"
)

CACHE_PATH = os.path.join(
    CACHE_ROOT,
    benchmark
)

os.makedirs(CACHE_PATH, exist_ok=True)

# ============================================================
# DCT TRANSFORMATION
# ============================================================

def compute_dct(image_path):

    """
    Same R1 representation:

    grayscale
    -> resize 224x224
    -> normalize
    -> orthonormal 2-D DCT
    -> absolute value
    -> log1p
    -> per-image min-max normalization
    -> 3 channels
    """

    image = tf.keras.utils.load_img(
        image_path,
        color_mode="grayscale",
        target_size=(IMG_SIZE, IMG_SIZE)
    )

    x = tf.keras.utils.img_to_array(
        image
    ).squeeze()

    x = x.astype(
        np.float32
    ) / 255.0

    coeff = dctn(
        x,
        type=2,
        norm="ortho"
    )

    coeff = np.abs(coeff)

    coeff = np.log1p(coeff)

    cmin = coeff.min()
    cmax = coeff.max()

    if cmax > cmin:

        coeff = (
            coeff - cmin
        ) / (
            cmax - cmin
        )

    else:

        coeff = np.zeros_like(
            coeff
        )

    # 3-channel representation
    coeff = np.stack(
        [coeff, coeff, coeff],
        axis=-1
    )

    return coeff.astype(
        np.float32
    )


# ============================================================
# CACHE CREATION
# ============================================================

def prepare_cache(split):

    split_path = os.path.join(
        DATASET_PATH,
        split
    )

    if split == "train":

        classes = [
            ("train_nhs", 0),
            ("train_hs", 1)
        ]

    else:

        classes = [
            ("test_nhs", 0),
            ("test_hs", 1)
        ]

    samples = []

    for class_name, label in classes:

        class_dir = os.path.join(
            split_path,
            class_name
        )

        if not os.path.isdir(class_dir):

            raise FileNotFoundError(
                f"Directory not found:\n"
                f"{class_dir}"
            )

        for filename in sorted(
            os.listdir(class_dir)
        ):

            if filename.lower().endswith(".png"):

                samples.append(
                    (
                        os.path.join(
                            class_dir,
                            filename
                        ),
                        label
                    )
                )

    cache_file = os.path.join(
        CACHE_PATH,
        f"{split}.npz"
    )

    # --------------------------------------------------------
    # Use existing cache
    # --------------------------------------------------------

    if os.path.exists(cache_file):

        print()
        print(
            f"Found existing {split} cache:"
        )

        print(cache_file)

        data = np.load(
            cache_file
        )

        X = data["X"]
        y = data["y"]

        print(
            f"Cached samples: {len(y)}"
        )

        return X, y

    # --------------------------------------------------------
    # Create cache
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print(
        f"CREATING DCT CACHE — {benchmark} — {split}"
    )
    print("=" * 75)

    print(
        f"Samples: {len(samples)}"
    )

    start = time.time()

    X = np.empty(
        (
            len(samples),
            IMG_SIZE,
            IMG_SIZE,
            3
        ),
        dtype=np.float32
    )

    y = np.empty(
        len(samples),
        dtype=np.int32
    )

    for i, (path, label) in enumerate(samples):

        X[i] = compute_dct(path)
        y[i] = label

        if (
            (i + 1) % 500 == 0
            or
            (i + 1) == len(samples)
        ):

            elapsed = time.time() - start

            print(
                f"{i + 1}/{len(samples)} "
                f"({elapsed:.1f}s)"
            )

    np.savez(
        cache_file,
        X=X,
        y=y
    )

    elapsed = time.time() - start

    print()
    print(
        f"Cache saved: {cache_file}"
    )

    print(
        f"Cache creation time: "
        f"{elapsed:.2f} sec"
    )

    return X, y


# ============================================================
# MODEL
# ============================================================

def build_model():

    """
    Parameter-matched R0-style CNN.

    The goal is to keep the classifier architecture
    controlled while changing only the representation.
    """

    model = models.Sequential([

        layers.Input(
            shape=(
                IMG_SIZE,
                IMG_SIZE,
                3
            )
        ),

        # ----------------------------------------------------
        # Block 1
        # ----------------------------------------------------

        layers.Conv2D(
            12,
            (3, 3),
            activation="elu"
        ),

        layers.Conv2D(
            12,
            (3, 3),
            activation="elu"
        ),

        layers.Conv2D(
            12,
            (3, 3),
            activation=None
        ),

        layers.BatchNormalization(
            momentum=0.99,
            epsilon=0.001
        ),

        layers.ELU(),

        layers.MaxPooling2D(
            (2, 2)
        ),

        # ----------------------------------------------------
        # Block 2
        # ----------------------------------------------------

        layers.Conv2D(
            12,
            (3, 3),
            activation="elu"
        ),

        layers.Conv2D(
            12,
            (3, 3),
            activation="elu"
        ),

        layers.Conv2D(
            12,
            (3, 3),
            activation=None
        ),

        layers.BatchNormalization(
            momentum=0.99,
            epsilon=0.001
        ),

        layers.ELU(),

        layers.MaxPooling2D(
            (2, 2)
        ),

        # ----------------------------------------------------
        # Final pooling
        # ----------------------------------------------------

        layers.MaxPooling2D(
            (5, 5)
        ),

        layers.Flatten(),

        layers.Dropout(
            0.3
        ),

        layers.Dense(
            10
        ),

        layers.Dense(
            1,
            activation="sigmoid"
        )
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Nadam(),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# MAIN
# ============================================================

print()
print("=" * 75)
print(
    f"CACHED DCT CNN — {benchmark}"
)
print("=" * 75)

print(
    f"Dataset: {dataset_name}"
)

print(
    f"Image size: {IMG_SIZE}x{IMG_SIZE}"
)

print(
    f"Epochs: {EPOCHS}"
)

print(
    f"Batch size: {BATCH_SIZE}"
)

print("=" * 75)


# ============================================================
# PREPARE DATA
# ============================================================

X_train, y_train = prepare_cache(
    "train"
)

X_test, y_test = prepare_cache(
    "test"
)

print()
print("=" * 75)
print("DATA READY")
print("=" * 75)

print(
    f"Training samples: {len(y_train)}"
)

print(
    f"Test samples: {len(y_test)}"
)


# ============================================================
# MODEL
# ============================================================

model = build_model()

print()
print(
    f"Trainable parameters: "
    f"{model.count_params():,}"
)

# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 75)
print("TRAINING")
print("=" * 75)

train_start = time.time()

history = model.fit(
    X_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    shuffle=True,
    verbose=1
)

train_time = (
    time.time() -
    train_start
)

# ============================================================
# INFERENCE
# ============================================================

print()
print("=" * 75)
print("INFERENCE")
print("=" * 75)

inference_start = time.time()

probabilities = model.predict(
    X_test,
    batch_size=BATCH_SIZE,
    verbose=1
).reshape(-1)

inference_time = (
    time.time() -
    inference_start
)

# ============================================================
# CLASSIFICATION
# ============================================================

predicted_labels = (
    probabilities >= 0.5
).astype(int)

cm = confusion_matrix(
    y_test,
    predicted_labels,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

precision = precision_score(
    y_test,
    predicted_labels,
    zero_division=0
)

recall = recall_score(
    y_test,
    predicted_labels,
    zero_division=0
)

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0.0
)

f1 = f1_score(
    y_test,
    predicted_labels,
    zero_division=0
)

balanced_accuracy = (
    recall + specificity
) / 2.0


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 75)
print(
    f"RESULTS — CACHED DCT CNN — {benchmark}"
)
print("=" * 75)

print(
    f"Balanced Accuracy : "
    f"{balanced_accuracy:.4f}"
)

print(
    f"Precision         : "
    f"{precision:.4f}"
)

print(
    f"Recall/Sensitivity: "
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

print()

print("Confusion Matrix:")
print()

print(
    "                 Pred NHS    Pred HS"
)

print(
    f"Actual NHS       "
    f"{tn:9d}    {fp:8d}"
)

print(
    f"Actual HS        "
    f"{fn:9d}    {tp:8d}"
)

print()

print(
    f"Training time    : "
    f"{train_time:.2f} sec"
)

print(
    f"Inference time   : "
    f"{inference_time:.2f} sec"
)

print(
    f"Test samples     : "
    f"{len(y_test)}"
)

print(
    f"Parameters       : "
    f"{model.count_params():,}"
)

print("=" * 75)


# ============================================================
# SAVE RESULTS
# ============================================================

result_file = os.path.join(
    CACHE_PATH,
    f"dct_cached_results_{benchmark}.txt"
)

with open(
    result_file,
    "w"
) as f:

    f.write(
        f"CACHED DCT CNN — {benchmark}\n"
    )

    f.write("=" * 60 + "\n")

    f.write(
        f"Balanced Accuracy : "
        f"{balanced_accuracy:.6f}\n"
    )

    f.write(
        f"Precision         : "
        f"{precision:.6f}\n"
    )

    f.write(
        f"Recall            : "
        f"{recall:.6f}\n"
    )

    f.write(
        f"Specificity       : "
        f"{specificity:.6f}\n"
    )

    f.write(
        f"F1                : "
        f"{f1:.6f}\n"
    )

    f.write("\n")

    f.write(
        f"TN={tn}, "
        f"FP={fp}, "
        f"FN={fn}, "
        f"TP={tp}\n"
    )

    f.write(
        f"\nTraining time: "
        f"{train_time:.2f} sec\n"
    )

    f.write(
        f"Inference time: "
        f"{inference_time:.2f} sec\n"
    )

    f.write(
        f"Parameters: "
        f"{model.count_params()}\n"
    )

print()
print(
    f"Results saved to:\n{result_file}"
)

print()
