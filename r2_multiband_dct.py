import os
import sys
import time

import numpy as np
import tensorflow as tf

from scipy.fft import dctn
from PIL import Image

from tensorflow.keras import layers

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

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10

TARGET_PARAMS = 12873

DATA_ROOT = os.path.expanduser(
    "~/Downloads/iccad-official"
)

CACHE_ROOT = os.path.expanduser(
    "~/Downloads/litho_r2_multiband_cache"
)

# Radial frequency boundaries
LOW_CUTOFF = 0.15
MID_CUTOFF = 0.40


# ============================================================
# BENCHMARK MAP
# ============================================================

BENCHMARK_MAP = {
    "B1": "iccad1",
    "B2": "iccad2",
    "B3": "iccad3",
    "B4": "iccad4",
    "B5": "iccad5"
}


# ============================================================
# DATASET PATHS
# ============================================================

def get_paths(benchmark):

    folder = BENCHMARK_MAP[benchmark]

    root = os.path.join(
        DATA_ROOT,
        folder
    )

    train_hs = os.path.join(
        root,
        "train",
        "train_hs"
    )

    train_nhs = os.path.join(
        root,
        "train",
        "train_nhs"
    )

    test_hs = os.path.join(
        root,
        "test",
        "test_hs"
    )

    test_nhs = os.path.join(
        root,
        "test",
        "test_nhs"
    )

    return (
        train_hs,
        train_nhs,
        test_hs,
        test_nhs
    )


# ============================================================
# IMAGE FILES
# ============================================================

def get_files(folder):

    files = []

    for f in os.listdir(folder):

        if f.lower().endswith(
            (".png", ".jpg", ".jpeg")
        ):
            files.append(
                os.path.join(folder, f)
            )

    files.sort()

    return files


# ============================================================
# PRECOMPUTE RADIAL FREQUENCY MASKS
# ============================================================

def create_frequency_masks():

    y = np.arange(
        IMG_SIZE
    )

    x = np.arange(
        IMG_SIZE
    )

    yy, xx = np.meshgrid(
        y,
        x,
        indexing="ij"
    )

    # Normalize coordinates so that
    # DC = (0,0) and maximum radial
    # frequency is approximately 1.

    radius = np.sqrt(
        (yy / (IMG_SIZE - 1)) ** 2
        +
        (xx / (IMG_SIZE - 1)) ** 2
    )

    # Normalize maximum radius to 1
    radius = radius / np.sqrt(2.0)

    low_mask = (
        radius <= LOW_CUTOFF
    )

    mid_mask = (
        (radius > LOW_CUTOFF)
        &
        (radius <= MID_CUTOFF)
    )

    high_mask = (
        radius > MID_CUTOFF
    )

    print()
    print("Frequency bands:")
    print(
        f"LOW  : r <= {LOW_CUTOFF}"
    )
    print(
        f"MID  : {LOW_CUTOFF} < r <= {MID_CUTOFF}"
    )
    print(
        f"HIGH : r > {MID_CUTOFF}"
    )

    print()
    print(
        "Band coefficients:"
    )

    print(
        f"LOW  : {low_mask.sum():,}"
    )

    print(
        f"MID  : {mid_mask.sum():,}"
    )

    print(
        f"HIGH : {high_mask.sum():,}"
    )

    return (
        low_mask,
        mid_mask,
        high_mask
    )


# ============================================================
# MULTI-BAND DCT TRANSFORM
#
# Output:
#
# H x W x 3
#
# Channel 0 = LOW
# Channel 1 = MID
# Channel 2 = HIGH
# ============================================================

def multiband_dct_transform(
    path,
    low_mask,
    mid_mask,
    high_mask
):

    img = Image.open(
        path
    ).convert("L")

    img = img.resize(
        (IMG_SIZE, IMG_SIZE),
        Image.Resampling.BILINEAR
    )

    img = np.asarray(
        img,
        dtype=np.float32
    )

    img = img / 255.0

    # --------------------------------------------------------
    # 2D orthonormal DCT
    # --------------------------------------------------------

    coeff = dctn(
        img,
        type=2,
        norm="ortho"
    )

    # Magnitude
    coeff = np.abs(
        coeff
    )

    # Log compression
    coeff = np.log1p(
        coeff
    )

    # --------------------------------------------------------
    # Normalize the COMPLETE DCT representation first.
    #
    # This preserves relative energy differences between
    # LOW/MID/HIGH bands.
    # --------------------------------------------------------

    cmin = coeff.min()
    cmax = coeff.max()

    if cmax > cmin:

        coeff = (
            coeff - cmin
        ) / (
            cmax - cmin
        )

    else:

        coeff.fill(0)

    # --------------------------------------------------------
    # Create frequency-band channels
    # --------------------------------------------------------

    low = np.zeros_like(
        coeff
    )

    mid = np.zeros_like(
        coeff
    )

    high = np.zeros_like(
        coeff
    )

    low[low_mask] = (
        coeff[low_mask]
    )

    mid[mid_mask] = (
        coeff[mid_mask]
    )

    high[high_mask] = (
        coeff[high_mask]
    )

    # --------------------------------------------------------
    # Stack into 3-channel representation
    # --------------------------------------------------------

    result = np.stack(
        [
            low,
            mid,
            high
        ],
        axis=-1
    )

    return result.astype(
        np.float16
    )


# ============================================================
# DISK-BACKED CACHE
#
# We store float16 HxWx3 arrays.
#
# This avoids loading the complete B2/B3 dataset into RAM.
# ============================================================

def create_cache(
    files,
    labels,
    cache_name,
    low_mask,
    mid_mask,
    high_mask
):

    os.makedirs(
        CACHE_ROOT,
        exist_ok=True
    )

    x_path = os.path.join(
        CACHE_ROOT,
        cache_name + "_X.npy"
    )

    y_path = os.path.join(
        CACHE_ROOT,
        cache_name + "_y.npy"
    )

    # --------------------------------------------------------
    # Existing cache
    # --------------------------------------------------------

    if (
        os.path.exists(x_path)
        and
        os.path.exists(y_path)
    ):

        print()
        print(
            f"Using existing cache: "
            f"{cache_name}"
        )

        X = np.load(
            x_path,
            mmap_mode="r"
        )

        y = np.load(
            y_path,
            mmap_mode="r"
        )

        return X, y

    # --------------------------------------------------------
    # Create cache
    # --------------------------------------------------------

    print()
    print(
        f"Creating Multi-Band DCT cache: "
        f"{cache_name}"
    )

    print(
        f"Images: {len(files):,}"
    )

    X = np.lib.format.open_memmap(
        x_path,
        mode="w+",
        dtype=np.float16,
        shape=(
            len(files),
            IMG_SIZE,
            IMG_SIZE,
            3
        )
    )

    y = np.lib.format.open_memmap(
        y_path,
        mode="w+",
        dtype=np.uint8,
        shape=(
            len(files),
        )
    )

    start = time.time()

    for i, path in enumerate(files):

        X[i] = multiband_dct_transform(
            path,
            low_mask,
            mid_mask,
            high_mask
        )

        y[i] = labels[i]

        if (
            (i + 1) % 250 == 0
            or
            i == len(files) - 1
        ):

            elapsed = (
                time.time()
                -
                start
            )

            print(
                f"\rProcessed "
                f"{i + 1:,}/"
                f"{len(files):,} "
                f"({100*(i+1)/len(files):.1f}%) "
                f"[{elapsed:.1f}s]",
                end=""
            )

    print()

    X.flush()
    y.flush()

    del X
    del y

    print(
        f"Cache written: {cache_name}"
    )

    X = np.load(
        x_path,
        mmap_mode="r"
    )

    y = np.load(
        y_path,
        mmap_mode="r"
    )

    return X, y


# ============================================================
# KERAS DATA SEQUENCE
# ============================================================

class MultiBandSequence(
    tf.keras.utils.Sequence
):

    def __init__(
        self,
        X,
        y,
        batch_size=32,
        shuffle=False
    ):

        super().__init__()

        self.X = X
        self.y = y

        self.batch_size = (
            batch_size
        )

        self.shuffle = shuffle

        self.indices = np.arange(
            len(y)
        )

        self.on_epoch_end()

    def __len__(self):

        return int(
            np.ceil(
                len(self.y)
                /
                self.batch_size
            )
        )

    def __getitem__(
        self,
        index
    ):

        start = (
            index *
            self.batch_size
        )

        end = min(
            start +
            self.batch_size,
            len(self.y)
        )

        idx = self.indices[
            start:end
        ]

        batch = np.asarray(
            self.X[idx],
            dtype=np.float32
        )

        labels = np.asarray(
            self.y[idx],
            dtype=np.float32
        )

        return (
            batch,
            labels
        )

    def on_epoch_end(self):

        if self.shuffle:

            np.random.shuffle(
                self.indices
            )


# ============================================================
# CNN
#
# EXACT SAME ARCHITECTURE AS R0/R1
#
# 12,873 PARAMETERS
# ============================================================

def build_model():

    inputs = layers.Input(
        shape=(
            IMG_SIZE,
            IMG_SIZE,
            3
        )
    )

    # --------------------------------------------------------
    # BLOCK 1
    # --------------------------------------------------------

    x = layers.Conv2D(
        12,
        (3, 3),
        activation="elu"
    )(inputs)

    x = layers.Conv2D(
        12,
        (3, 3),
        activation="elu"
    )(x)

    x = layers.Conv2D(
        12,
        (3, 3),
        activation=None
    )(x)

    x = layers.BatchNormalization(
        momentum=0.99,
        epsilon=0.001
    )(x)

    x = layers.ELU()(x)

    x = layers.MaxPooling2D(
        (2, 2)
    )(x)

    # Critical 5x5 pooling
    x = layers.MaxPooling2D(
        (5, 5)
    )(x)

    # --------------------------------------------------------
    # BLOCK 2
    # --------------------------------------------------------

    x = layers.Conv2D(
        12,
        (3, 3),
        activation="elu"
    )(x)

    x = layers.Conv2D(
        12,
        (3, 3),
        activation="elu"
    )(x)

    x = layers.Conv2D(
        12,
        (3, 3),
        activation=None
    )(x)

    x = layers.BatchNormalization(
        momentum=0.99,
        epsilon=0.001
    )(x)

    x = layers.ELU()(x)

    x = layers.MaxPooling2D(
        (2, 2)
    )(x)

    # --------------------------------------------------------
    # CLASSIFIER
    # --------------------------------------------------------

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

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs
    )

    model.compile(
        optimizer=tf.keras.optimizers.Nadam(),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# SPECIFICITY
# ============================================================

def calculate_specificity(
    y_true,
    y_pred
):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    if tn + fp == 0:

        return 0.0

    return (
        tn /
        (tn + fp)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) < 2:

        print(
            "Usage:"
        )

        print(
            "python r2_multiband_dct.py B1"
        )

        print(
            "python r2_multiband_dct.py B2"
        )

        print(
            "python r2_multiband_dct.py B3"
        )

        print(
            "python r2_multiband_dct.py B4"
        )

        print(
            "python r2_multiband_dct.py B5"
        )

        sys.exit(1)

    benchmark = (
        sys.argv[1]
        .upper()
    )

    if benchmark not in BENCHMARK_MAP:

        print(
            "Invalid benchmark."
        )

        sys.exit(1)

    print()
    print("=" * 70)

    print(
        f"R2 MULTI-BAND DCT — {benchmark}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # FREQUENCY MASKS
    # --------------------------------------------------------

    (
        low_mask,
        mid_mask,
        high_mask
    ) = create_frequency_masks()

    # --------------------------------------------------------
    # PATHS
    # --------------------------------------------------------

    (
        train_hs,
        train_nhs,
        test_hs,
        test_nhs
    ) = get_paths(
        benchmark
    )

    # --------------------------------------------------------
    # FILES
    # --------------------------------------------------------

    train_hs_files = get_files(
        train_hs
    )

    train_nhs_files = get_files(
        train_nhs
    )

    test_hs_files = get_files(
        test_hs
    )

    test_nhs_files = get_files(
        test_nhs
    )

    train_files = (
        train_hs_files
        +
        train_nhs_files
    )

    test_files = (
        test_hs_files
        +
        test_nhs_files
    )

    train_labels = (
        [1] * len(
            train_hs_files
        )
        +
        [0] * len(
            train_nhs_files
        )
    )

    test_labels = (
        [1] * len(
            test_hs_files
        )
        +
        [0] * len(
            test_nhs_files
        )
    )

    # --------------------------------------------------------
    # SHUFFLE TRAINING DATA
    # --------------------------------------------------------

    rng = np.random.default_rng(
        42
    )

    train_order = rng.permutation(
        len(train_files)
    )

    train_files = [
        train_files[i]
        for i in train_order
    ]

    train_labels = [
        train_labels[i]
        for i in train_order
    ]

    # --------------------------------------------------------
    # CACHE
    # --------------------------------------------------------

    X_train, y_train = create_cache(
        train_files,
        train_labels,
        benchmark + "_train",
        low_mask,
        mid_mask,
        high_mask
    )

    X_test, y_test = create_cache(
        test_files,
        test_labels,
        benchmark + "_test",
        low_mask,
        mid_mask,
        high_mask
    )

    print()
    print(
        f"Train samples: {len(y_train):,}"
    )

    print(
        f"Test samples : {len(y_test):,}"
    )

    # --------------------------------------------------------
    # DATA SEQUENCES
    # --------------------------------------------------------

    train_seq = MultiBandSequence(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_seq = MultiBandSequence(
        X_test,
        y_test,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = build_model()

    print()

    model.summary()

    print()
    print("=" * 70)

    print(
        f"PARAMETERS: "
        f"{model.count_params():,}"
    )

    print(
        f"TARGET    : "
        f"{TARGET_PARAMS:,}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # HARD PARAMETER CHECK
    # --------------------------------------------------------

    if (
        model.count_params()
        != TARGET_PARAMS
    ):

        raise RuntimeError(
            "Parameter mismatch! "
            f"Expected {TARGET_PARAMS:,}, "
            f"got {model.count_params():,}"
        )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    print()
    print(
        "Starting training..."
    )

    start_train = time.time()

    history = model.fit(
        train_seq,
        epochs=EPOCHS,
        verbose=1
    )

    train_time = (
        time.time()
        -
        start_train
    )

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    print()
    print(
        "Starting inference..."
    )

    start_inference = time.time()

    probabilities = model.predict(
        test_seq,
        verbose=1
    ).ravel()

    inference_time = (
        time.time()
        -
        start_inference
    )

    y_pred = (
        probabilities >= 0.5
    ).astype(int)

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    ba = balanced_accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    specificity = (
        calculate_specificity(
            y_test,
            y_pred
        )
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 70)

    print(
        f"FINAL RESULTS — R2 {benchmark}"
    )

    print("=" * 70)

    print(
        f"Balanced Accuracy : "
        f"{ba:.4f} "
        f"({ba*100:.2f}%)"
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

    print()

    print(
        "Confusion Matrix:"
    )

    print(
        "                 Pred NHS   Pred HS"
    )

    print(
        f"Actual NHS       "
        f"{cm[0,0]:8d} "
        f"{cm[0,1]:9d}"
    )

    print(
        f"Actual HS        "
        f"{cm[1,0]:8d} "
        f"{cm[1,1]:9d}"
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
        f"Parameters       : "
        f"{model.count_params():,}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    results_file = os.path.join(
        CACHE_ROOT,
        benchmark +
        "_results.txt"
    )

    with open(
        results_file,
        "w"
    ) as f:

        f.write(
            f"Experiment: R2 Multi-Band DCT\n"
        )

        f.write(
            f"Benchmark: {benchmark}\n"
        )

        f.write(
            f"Parameters: "
            f"{model.count_params()}\n"
        )

        f.write(
            f"Low cutoff: "
            f"{LOW_CUTOFF}\n"
        )

        f.write(
            f"Mid cutoff: "
            f"{MID_CUTOFF}\n"
        )

        f.write(
            f"Balanced Accuracy: "
            f"{ba:.6f}\n"
        )

        f.write(
            f"Precision: "
            f"{precision:.6f}\n"
        )

        f.write(
            f"Recall: "
            f"{recall:.6f}\n"
        )

        f.write(
            f"Specificity: "
            f"{specificity:.6f}\n"
        )

        f.write(
            f"F1: "
            f"{f1:.6f}\n"
        )

        f.write(
            f"Training Time: "
            f"{train_time:.2f}\n"
        )

        f.write(
            f"Inference Time: "
            f"{inference_time:.2f}\n"
        )

        f.write(
            "\nConfusion Matrix:\n"
        )

        f.write(
            str(cm)
        )

    print()

    print(
        "Results saved to:"
    )

    print(
        results_file
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
