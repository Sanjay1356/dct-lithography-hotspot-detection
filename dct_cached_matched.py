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
    "~/Downloads/litho_dct_cache_matched"
)

# ============================================================
# BENCHMARK PATHS
# ============================================================

BENCHMARK_MAP = {
    "B1": "iccad1",
    "B2": "iccad2",
    "B3": "iccad3",
    "B4": "iccad4",
    "B5": "iccad5",
}


# ============================================================
# DATASET PATH
# ============================================================

def get_paths(benchmark):

    folder = BENCHMARK_MAP[benchmark]

    root = os.path.join(
        DATA_ROOT,
        folder
    )

    train_hs = os.path.join(
        root, "train", "train_hs"
    )

    train_nhs = os.path.join(
        root, "train", "train_nhs"
    )

    test_hs = os.path.join(
        root, "test", "test_hs"
    )

    test_nhs = os.path.join(
        root, "test", "test_nhs"
    )

    return (
        train_hs,
        train_nhs,
        test_hs,
        test_nhs
    )


# ============================================================
# GET IMAGE FILES
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
# DCT TRANSFORMATION
# ============================================================

def dct_transform(path):

    img = Image.open(path).convert("L")

    img = img.resize(
        (IMG_SIZE, IMG_SIZE),
        Image.Resampling.BILINEAR
    )

    img = np.asarray(
        img,
        dtype=np.float32
    )

    # Normalize image
    img = img / 255.0

    # 2D orthonormal DCT
    coeff = dctn(
        img,
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

        coeff.fill(0)

    return coeff.astype(
        np.float16
    )


# ============================================================
# CACHE CREATION
#
# IMPORTANT:
# We cache ONLY ONE DCT CHANNEL as float16.
# During training it is replicated to 3 channels.
#
# This prevents B2/B3 from consuming enormous RAM.
# ============================================================

def create_cache(
    files,
    labels,
    cache_name
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

    if (
        os.path.exists(x_path)
        and
        os.path.exists(y_path)
    ):

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

    print()
    print(
        f"Creating DCT cache: "
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
            IMG_SIZE
        )
    )

    y = np.lib.format.open_memmap(
        y_path,
        mode="w+",
        dtype=np.uint8,
        shape=(len(files),)
    )

    start = time.time()

    for i, path in enumerate(files):

        X[i] = dct_transform(path)

        y[i] = labels[i]

        if (
            (i + 1) % 250 == 0
            or
            i == len(files) - 1
        ):

            elapsed = time.time() - start

            print(
                f"\rProcessed "
                f"{i+1:,}/{len(files):,} "
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
# KERAS SEQUENCE
#
# Reads only one batch from the disk-backed cache.
# ============================================================

class DCTSequence(
    tf.keras.utils.Sequence
):

    def __init__(
        self,
        X,
        y,
        batch_size=32,
        shuffle=False
    ):

        self.X = X
        self.y = y

        self.batch_size = batch_size
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

    def __getitem__(self, index):

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

        # Read one-channel DCT
        batch = np.asarray(
            self.X[idx],
            dtype=np.float32
        )

        # Convert:
        #
        # B x H x W
        #
        # into:
        #
        # B x H x W x 3
        #
        batch = np.repeat(
            batch[..., np.newaxis],
            3,
            axis=-1
        )

        labels = np.asarray(
            self.y[idx],
            dtype=np.float32
        )

        return batch, labels

    def on_epoch_end(self):

        if self.shuffle:

            np.random.shuffle(
                self.indices
            )


# ============================================================
# EXACT R0 ARCHITECTURE
#
# The critical detail:
#
# Block 1
#    Conv
#    Conv
#    Conv
#    BN
#    ELU
#    MaxPool 2x2
#
# Extra MaxPool 5x5
#
# Block 2
#    Conv
#    Conv
#    Conv
#    BN
#    ELU
#    MaxPool 2x2
#
# Flatten
# Dense
# Dense
#
# This gives exactly 12,873 parameters.
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

    # --------------------------------------------------------
    # IMPORTANT 5x5 POOL
    #
    # This is BETWEEN the two convolutional blocks.
    # This is what gives the original 12,873 parameter
    # architecture rather than 18,993.
    # --------------------------------------------------------

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

    if (
        tn + fp
        ==
        0
    ):

        return 0.0

    return tn / (
        tn + fp
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
            "python dct_cached_matched.py B1"
        )

        print(
            "python dct_cached_matched.py B2"
        )

        print(
            "python dct_cached_matched.py B3"
        )

        print(
            "python dct_cached_matched.py B4"
        )

        print(
            "python dct_cached_matched.py B5"
        )

        sys.exit(1)

    benchmark = sys.argv[1].upper()

    if benchmark not in BENCHMARK_MAP:

        print(
            "Invalid benchmark."
        )

        sys.exit(1)

    print()
    print("=" * 70)
    print(
        f"DCT CNN — PARAMETER-MATCHED EXPERIMENT — {benchmark}"
    )
    print("=" * 70)

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
        train_hs_files +
        train_nhs_files
    )

    test_files = (
        test_hs_files +
        test_nhs_files
    )

    train_labels = (
        [1] *
        len(train_hs_files)
        +
        [0] *
        len(train_nhs_files)
    )

    test_labels = (
        [1] *
        len(test_hs_files)
        +
        [0] *
        len(test_nhs_files)
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
        benchmark + "_train"
    )

    X_test, y_test = create_cache(
        test_files,
        test_labels,
        benchmark + "_test"
    )

    print()
    print(
        f"Train samples: {len(y_train):,}"
    )

    print(
        f"Test samples : {len(y_test):,}"
    )

    # --------------------------------------------------------
    # SEQUENCES
    # --------------------------------------------------------

    train_seq = DCTSequence(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_seq = DCTSequence(
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
    print(
        "=" * 70
    )

    print(
        f"PARAMETERS: {model.count_params():,}"
    )

    print(
        f"TARGET    : {TARGET_PARAMS:,}"
    )

    print(
        "=" * 70
    )

    if model.count_params() != TARGET_PARAMS:

        raise RuntimeError(
            f"Parameter mismatch! "
            f"Expected {TARGET_PARAMS:,}, "
            f"got {model.count_params():,}"
        )

    # --------------------------------------------------------
    # TRAIN
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

    specificity = calculate_specificity(
        y_test,
        y_pred
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
        f"FINAL RESULTS — {benchmark}"
    )
    print("=" * 70)

    print(
        f"Balanced Accuracy : {ba:.4f} "
        f"({ba*100:.2f}%)"
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
    print(
        "Confusion Matrix:"
    )

    print(
        "                 Pred NHS   Pred HS"
    )

    print(
        f"Actual NHS       {cm[0,0]:8d} "
        f"{cm[0,1]:9d}"
    )

    print(
        f"Actual HS        {cm[1,0]:8d} "
        f"{cm[1,1]:9d}"
    )

    print()
    print(
        f"Training time    : {train_time:.2f} sec"
    )

    print(
        f"Inference time   : {inference_time:.2f} sec"
    )

    print(
        f"Parameters       : {model.count_params():,}"
    )

    print(
        "=" * 70
    )

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
            f"Benchmark: {benchmark}\n"
        )

        f.write(
            f"Parameters: {model.count_params()}\n"
        )

        f.write(
            f"Balanced Accuracy: {ba:.6f}\n"
        )

        f.write(
            f"Precision: {precision:.6f}\n"
        )

        f.write(
            f"Recall: {recall:.6f}\n"
        )

        f.write(
            f"Specificity: {specificity:.6f}\n"
        )

        f.write(
            f"F1: {f1:.6f}\n"
        )

        f.write(
            f"Training Time: {train_time:.2f}\n"
        )

        f.write(
            f"Inference Time: {inference_time:.2f}\n"
        )

        f.write(
            "\nConfusion Matrix:\n"
        )

        f.write(
            str(cm)
        )

    print()
    print(
        f"Results saved to:"
    )

    print(
        results_file
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
