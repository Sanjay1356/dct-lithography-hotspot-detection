import os
import sys
import time
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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
# CHECK BENCHMARK
# ============================================================

if len(sys.argv) != 2:

    print()
    print("Usage:")
    print("python dct_model.py B1")
    print("python dct_model.py B2")
    print("python dct_model.py B3")
    print("python dct_model.py B4")
    print("python dct_model.py B5")
    print()

    sys.exit(1)

benchmark = sys.argv[1].upper()

if benchmark not in BENCHMARK_MAP:

    print(
        f"Invalid benchmark: {benchmark}"
    )

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

# ============================================================
# DCT PREPROCESSING
# ============================================================

from scipy.fft import dctn


def dct_transform(image):

    """
    Convert spatial image to a DCT representation.

    Steps:
    1. grayscale
    2. resize to 224x224
    3. normalize
    4. 2-D orthonormal DCT
    5. absolute magnitude
    6. log compression
    7. per-image min-max normalization
    8. replicate to 3 channels
    """

    image = tf.image.rgb_to_grayscale(image)

    image = tf.image.resize(
        image,
        [IMG_SIZE, IMG_SIZE]
    )

    image = tf.cast(
        image,
        tf.float32
    ) / 255.0

    # Convert TensorFlow tensor to NumPy
    x = image.numpy().squeeze()

    # 2-D orthonormal DCT
    coeff = dctn(
        x,
        type=2,
        norm="ortho"
    )

    # Magnitude
    coeff = np.abs(coeff)

    # Log compression
    coeff = np.log1p(coeff)

    # Per-image normalization
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

    # Replicate to 3 channels
    coeff = np.stack(
        [coeff, coeff, coeff],
        axis=-1
    )

    return coeff.astype(
        np.float32
    )


# ============================================================
# DATA GENERATOR
# ============================================================

class DCTSequence(
    tf.keras.utils.Sequence
):

    def __init__(
        self,
        directory,
        batch_size=32,
        shuffle=False
    ):

        self.directory = directory

        self.batch_size = batch_size

        self.shuffle = shuffle

        self.samples = []

        self.class_names = [
            "train_hs",
            "train_nhs"
        ]

        # ----------------------------------------------------
        # IMPORTANT:
        # HS = 1
        # NHS = 0
        # ----------------------------------------------------

        for class_name in self.class_names:

            class_dir = os.path.join(
                directory,
                class_name
            )

            if not os.path.isdir(
                class_dir
            ):

                raise FileNotFoundError(
                    f"Directory not found:\n"
                    f"{class_dir}"
                )

            label = (
                1
                if class_name == "train_hs"
                else 0
            )

            for filename in os.listdir(
                class_dir
            ):

                if filename.lower().endswith(
                    ".png"
                ):

                    self.samples.append(
                        (
                            os.path.join(
                                class_dir,
                                filename
                            ),
                            label
                        )
                    )

        self.indices = np.arange(
            len(self.samples)
        )

        self.on_epoch_end()

    def __len__(self):

        return int(
            np.ceil(
                len(self.samples) /
                self.batch_size
            )
        )

    def __getitem__(self, index):

        batch_indices = self.indices[
            index *
            self.batch_size:
            (index + 1) *
            self.batch_size
        ]

        batch_x = []
        batch_y = []

        for idx in batch_indices:

            path, label = self.samples[
                idx
            ]

            image = tf.keras.utils.load_img(
                path,
                color_mode="rgb"
            )

            image = tf.keras.utils.img_to_array(
                image
            )

            dct_image = dct_transform(
                tf.convert_to_tensor(
                    image
                )
            )

            batch_x.append(
                dct_image
            )

            batch_y.append(
                label
            )

        return (
            np.asarray(
                batch_x,
                dtype=np.float32
            ),
            np.asarray(
                batch_y,
                dtype=np.float32
            )
        )

    def on_epoch_end(self):

        if self.shuffle:

            np.random.shuffle(
                self.indices
            )


# ============================================================
# TEST GENERATOR
# ============================================================

class DCTTestSequence(
    tf.keras.utils.Sequence
):

    def __init__(
        self,
        directory,
        batch_size=32
    ):

        self.directory = directory

        self.batch_size = batch_size

        self.samples = []

        # ----------------------------------------------------
        # TEST:
        # test_hs  = 1
        # test_nhs = 0
        # ----------------------------------------------------

        classes = [
            ("test_nhs", 0),
            ("test_hs", 1)
        ]

        for class_name, label in classes:

            class_dir = os.path.join(
                directory,
                class_name
            )

            if not os.path.isdir(
                class_dir
            ):

                raise FileNotFoundError(
                    f"Directory not found:\n"
                    f"{class_dir}"
                )

            for filename in os.listdir(
                class_dir
            ):

                if filename.lower().endswith(
                    ".png"
                ):

                    self.samples.append(
                        (
                            os.path.join(
                                class_dir,
                                filename
                            ),
                            label
                        )
                    )

        self.indices = np.arange(
            len(self.samples)
        )

    def __len__(self):

        return int(
            np.ceil(
                len(self.samples) /
                self.batch_size
            )
        )

    def __getitem__(self, index):

        batch_indices = self.indices[
            index *
            self.batch_size:
            (index + 1) *
            self.batch_size
        ]

        batch_x = []
        batch_y = []

        for idx in batch_indices:

            path, label = self.samples[
                idx
            ]

            image = tf.keras.utils.load_img(
                path,
                color_mode="rgb"
            )

            image = tf.keras.utils.img_to_array(
                image
            )

            dct_image = dct_transform(
                tf.convert_to_tensor(
                    image
                )
            )

            batch_x.append(
                dct_image
            )

            batch_y.append(
                label
            )

        return (
            np.asarray(
                batch_x,
                dtype=np.float32
            ),
            np.asarray(
                batch_y,
                dtype=np.float32
            )
        )


# ============================================================
# MODEL
# ============================================================

def build_model():

    model = models.Sequential(
        [

            layers.Input(
                shape=(
                    IMG_SIZE,
                    IMG_SIZE,
                    3
                )
            ),

            # ----------------------------
            # Block 1
            # ----------------------------

            layers.Conv2D(
                12,
                (3, 3),
                activation="elu",
                padding="valid"
            ),

            layers.Conv2D(
                12,
                (3, 3),
                activation="elu",
                padding="valid"
            ),

            layers.Conv2D(
                12,
                (3, 3),
                activation=None,
                padding="valid"
            ),

            layers.BatchNormalization(
                momentum=0.99,
                epsilon=0.001
            ),

            layers.ELU(),

            layers.MaxPooling2D(
                pool_size=(2, 2)
            ),

            # ----------------------------
            # Block 2
            # ----------------------------

            layers.Conv2D(
                12,
                (3, 3),
                activation="elu",
                padding="valid"
            ),

            layers.Conv2D(
                12,
                (3, 3),
                activation="elu",
                padding="valid"
            ),

            layers.Conv2D(
                12,
                (3, 3),
                activation=None,
                padding="valid"
            ),

            layers.BatchNormalization(
                momentum=0.99,
                epsilon=0.001
            ),

            layers.ELU(),

            layers.MaxPooling2D(
                pool_size=(2, 2)
            ),

            # ----------------------------
            # Final pooling
            # ----------------------------

            layers.MaxPooling2D(
                pool_size=(5, 5)
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
        ]
    )

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
print(f"DCT CNN — {benchmark}")
print("=" * 75)

print(
    f"Dataset: {dataset_name}"
)

print(
    f"Train path: {TRAIN_PATH}"
)

print(
    f"Test path : {TEST_PATH}"
)

print("=" * 75)
print()


# ============================================================
# LOAD DATA
# ============================================================

train_generator = DCTSequence(
    TRAIN_PATH,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_generator = DCTTestSequence(
    TEST_PATH,
    batch_size=BATCH_SIZE
)

print(
    f"Training samples : "
    f"{len(train_generator.samples)}"
)

print(
    f"Test samples     : "
    f"{len(test_generator.samples)}"
)

print()


# ============================================================
# MODEL
# ============================================================

model = build_model()

print(
    f"Trainable parameters: "
    f"{model.count_params():,}"
)

print()


# ============================================================
# TRAIN
# ============================================================

print("=" * 75)
print("TRAINING")
print("=" * 75)

train_start = time.time()

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    verbose=1
)

train_time = (
    time.time() -
    train_start
)

print()
print(
    f"Training time: "
    f"{train_time:.2f} seconds"
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
    test_generator,
    verbose=1
).reshape(-1)

inference_time = (
    time.time() -
    inference_start
)

# ------------------------------------------------------------
# IMPORTANT:
# The samples are stored in deterministic order:
# NHS first, then HS.
# ------------------------------------------------------------

true_labels = np.array([
    label
    for _, label
    in test_generator.samples
])

predicted_labels = (
    probabilities >= 0.5
).astype(int)


# ============================================================
# METRICS
# ============================================================

cm = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

precision = precision_score(
    true_labels,
    predicted_labels,
    zero_division=0
)

recall = recall_score(
    true_labels,
    predicted_labels,
    zero_division=0
)

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0
)

f1 = f1_score(
    true_labels,
    predicted_labels,
    zero_division=0
)

sensitivity = recall

balanced_accuracy = (
    sensitivity +
    specificity
) / 2


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 75)
print(f"RESULTS — DCT CNN — {benchmark}")
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
    f"{len(true_labels)}"
)

print(
    f"Parameters       : "
    f"{model.count_params():,}"
)

print("=" * 75)


# ============================================================
# SAVE RESULTS
# ============================================================

output_file = (
    f"dct_results_{benchmark}.txt"
)

with open(
    output_file,
    "w"
) as f:

    f.write(
        f"DCT CNN RESULTS — {benchmark}\n"
    )

    f.write(
        "=" * 60 + "\n"
    )

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

    f.write(
        "\nConfusion Matrix\n"
    )

    f.write(
        f"TN={tn}, FP={fp}, "
        f"FN={fn}, TP={tp}\n"
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
    f"Results saved to: "
    f"{output_file}"
)
print()
