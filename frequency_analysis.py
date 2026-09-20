import os
import sys
import csv
import numpy as np
from PIL import Image
from scipy.fft import dctn
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = os.path.expanduser("~/Downloads/iccad-official")

IMG_SIZE = 224

LOW_CUTOFF = 0.15
MID_CUTOFF = 0.40

BENCHMARK_MAP = {
    "B1": "iccad1",
    "B2": "iccad2",
    "B3": "iccad3",
    "B4": "iccad4",
    "B5": "iccad5"
}


# ============================================================
# DCT + SPATIAL FEATURE EXTRACTION
# ============================================================

def extract_features(image_path):

    img = Image.open(image_path).convert("L")
    img = img.resize(
        (IMG_SIZE, IMG_SIZE),
        Image.Resampling.BILINEAR
    )

    x = np.asarray(img, dtype=np.float32) / 255.0

    # --------------------------------------------------------
    # Spatial-domain features
    # --------------------------------------------------------

    spatial_mean = np.mean(x)
    spatial_variance = np.var(x)

    # --------------------------------------------------------
    # 2-D orthonormal DCT
    # --------------------------------------------------------

    coeff = dctn(
        x,
        type=2,
        norm="ortho"
    )

    energy = coeff ** 2

    h, w = energy.shape

    # Frequency coordinate grid
    u = np.arange(h)
    v = np.arange(w)

    U, V = np.meshgrid(
        u,
        v,
        indexing="ij"
    )

    # Radial normalized frequency
    radius = np.sqrt(U**2 + V**2)

    max_radius = np.sqrt(
        (h - 1)**2 +
        (w - 1)**2
    )

    r_norm = radius / max_radius

    # --------------------------------------------------------
    # Frequency bands
    # --------------------------------------------------------

    low_mask = r_norm <= LOW_CUTOFF

    mid_mask = (
        (r_norm > LOW_CUTOFF) &
        (r_norm <= MID_CUTOFF)
    )

    high_mask = r_norm > MID_CUTOFF

    # --------------------------------------------------------
    # Energy
    # --------------------------------------------------------

    total_energy = np.sum(energy) + 1e-12

    low_energy = np.sum(
        energy[low_mask]
    )

    mid_energy = np.sum(
        energy[mid_mask]
    )

    high_energy = np.sum(
        energy[high_mask]
    )

    # --------------------------------------------------------
    # Normalized spectral features
    # --------------------------------------------------------

    low_ratio = low_energy / total_energy

    mid_ratio = mid_energy / total_energy

    high_ratio = high_energy / total_energy

    high_low_ratio = (
        high_energy /
        (low_energy + 1e-12)
    )

    spectral_centroid = (
        np.sum(r_norm * energy) /
        total_energy
    )

    dc_energy_ratio = (
        energy[0, 0] /
        total_energy
    )

    return {
        "spatial_mean": spatial_mean,
        "spatial_variance": spatial_variance,

        "low_energy_ratio": low_ratio,
        "mid_energy_ratio": mid_ratio,
        "high_energy_ratio": high_ratio,
        "high_low_ratio": high_low_ratio,
        "spectral_centroid": spectral_centroid,
        "dc_energy_ratio": dc_energy_ratio,

        "total_dct_energy": total_energy
    }


# ============================================================
# LOAD DATASET
# ============================================================

def collect_images(benchmark):

    dataset_folder = BENCHMARK_MAP[benchmark]

    root = os.path.join(
        DATASET_ROOT,
        dataset_folder
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

    if not os.path.isdir(train_hs):
        raise FileNotFoundError(
            f"HS directory not found:\n{train_hs}"
        )

    if not os.path.isdir(train_nhs):
        raise FileNotFoundError(
            f"NHS directory not found:\n{train_nhs}"
        )

    hs_files = sorted([
        os.path.join(train_hs, f)
        for f in os.listdir(train_hs)
        if f.lower().endswith(".png")
    ])

    nhs_files = sorted([
        os.path.join(train_nhs, f)
        for f in os.listdir(train_nhs)
        if f.lower().endswith(".png")
    ])

    return hs_files, nhs_files


# ============================================================
# STATISTICAL ANALYSIS
# ============================================================

def analyze_feature(
    feature_name,
    hs_values,
    nhs_values
):

    hs_values = np.asarray(
        hs_values,
        dtype=np.float64
    )

    nhs_values = np.asarray(
        nhs_values,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Cohen's d
    # --------------------------------------------------------

    n_hs = len(hs_values)
    n_nhs = len(nhs_values)

    hs_var = np.var(
        hs_values,
        ddof=1
    )

    nhs_var = np.var(
        nhs_values,
        ddof=1
    )

    pooled_std = np.sqrt(
        (
            (n_hs - 1) * hs_var +
            (n_nhs - 1) * nhs_var
        )
        /
        (
            n_hs + n_nhs - 2
        )
    )

    if pooled_std > 0:

        cohens_d = (
            np.mean(hs_values) -
            np.mean(nhs_values)
        ) / pooled_std

    else:

        cohens_d = 0.0

    # --------------------------------------------------------
    # AUC
    # --------------------------------------------------------

    labels = np.concatenate([
        np.ones(n_hs),
        np.zeros(n_nhs)
    ])

    values = np.concatenate([
        hs_values,
        nhs_values
    ])

    try:

        auc = roc_auc_score(
            labels,
            values
        )

        # Direction-independent separation
        auc_abs = max(
            auc,
            1.0 - auc
        )

    except Exception:

        auc = 0.5
        auc_abs = 0.5

    # --------------------------------------------------------
    # Mann-Whitney U
    # --------------------------------------------------------

    try:

        _, p_value = mannwhitneyu(
            hs_values,
            nhs_values,
            alternative="two-sided"
        )

    except Exception:

        p_value = np.nan

    return {
        "feature": feature_name,

        "HS_mean": np.mean(hs_values),
        "HS_std": np.std(
            hs_values,
            ddof=1
        ),

        "NHS_mean": np.mean(nhs_values),
        "NHS_std": np.std(
            nhs_values,
            ddof=1
        ),

        "Cohens_d": cohens_d,

        "AUC": auc,
        "AUC_abs": auc_abs,

        "p_value": p_value
    }


# ============================================================
# MAIN
# ============================================================

if len(sys.argv) != 2:

    print()
    print("Usage:")
    print("  python frequency_analysis.py B1")
    print("  python frequency_analysis.py B2")
    print("  python frequency_analysis.py B3")
    print("  python frequency_analysis.py B4")
    print("  python frequency_analysis.py B5")
    print()

    sys.exit(1)


benchmark = sys.argv[1].upper()

if benchmark not in BENCHMARK_MAP:

    print(
        f"ERROR: Unknown benchmark '{benchmark}'"
    )

    sys.exit(1)


print()
print("=" * 75)
print(f"DCT FREQUENCY ANALYSIS — {benchmark}")
print("=" * 75)

print(
    f"Dataset: {BENCHMARK_MAP[benchmark]}"
)

print(
    f"Image size: {IMG_SIZE} × {IMG_SIZE}"
)

print(
    f"Frequency bands: "
    f"LOW ≤ {LOW_CUTOFF}, "
    f"MID ≤ {MID_CUTOFF}, "
    f"HIGH > {MID_CUTOFF}"
)

print("=" * 75)
print()


# ============================================================
# COLLECT FILES
# ============================================================

hs_files, nhs_files = collect_images(
    benchmark
)

print(
    f"HS images  : {len(hs_files)}"
)

print(
    f"NHS images : {len(nhs_files)}"
)

print()


# ============================================================
# EXTRACT FEATURES
# ============================================================

hs_features = []

nhs_features = []


print("Extracting HS features...")

for i, path in enumerate(hs_files):

    try:

        features = extract_features(
            path
        )

        hs_features.append(
            features
        )

    except Exception as e:

        print(
            f"ERROR: {path}"
        )

        print(e)

    if (
        (i + 1) % 100 == 0
        or
        (i + 1) == len(hs_files)
    ):

        print(
            f"  HS: "
            f"{i + 1}/{len(hs_files)}"
        )


print()
print("Extracting NHS features...")


for i, path in enumerate(nhs_files):

    try:

        features = extract_features(
            path
        )

        nhs_features.append(
            features
        )

    except Exception as e:

        print(
            f"ERROR: {path}"
        )

        print(e)

    if (
        (i + 1) % 1000 == 0
        or
        (i + 1) == len(nhs_files)
    ):

        print(
            f"  NHS: "
            f"{i + 1}/{len(nhs_files)}"
        )


# ============================================================
# ANALYZE FEATURES
# ============================================================

feature_names = [

    # Spatial controls
    "spatial_mean",
    "spatial_variance",

    # Frequency features
    "low_energy_ratio",
    "mid_energy_ratio",
    "high_energy_ratio",
    "high_low_ratio",
    "spectral_centroid",
    "dc_energy_ratio",

    # Total energy
    "total_dct_energy"
]


results = []


for feature_name in feature_names:

    hs_values = [
        x[feature_name]
        for x in hs_features
    ]

    nhs_values = [
        x[feature_name]
        for x in nhs_features
    ]

    result = analyze_feature(
        feature_name,
        hs_values,
        nhs_values
    )

    results.append(
        result
    )


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 115)
print(f"RESULTS — {benchmark}")
print("=" * 115)

header = (
    f"{'Feature':25s}"
    f"{'HS Mean':>14s}"
    f"{'NHS Mean':>14s}"
    f"{'Cohen d':>12s}"
    f"{'AUC':>12s}"
    f"{'p-value':>18s}"
)

print(header)

print("-" * 115)


for r in results:

    print(
        f"{r['feature']:25s}"
        f"{r['HS_mean']:14.6f}"
        f"{r['NHS_mean']:14.6f}"
        f"{r['Cohens_d']:12.4f}"
        f"{r['AUC_abs']:12.4f}"
        f"{r['p_value']:18.4e}"
    )


# ============================================================
# SAVE CSV
# ============================================================

output_file = (
    f"frequency_results_{benchmark}.csv"
)


with open(
    output_file,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "feature",
            "HS_mean",
            "HS_std",
            "NHS_mean",
            "NHS_std",
            "Cohens_d",
            "AUC",
            "AUC_abs",
            "p_value"
        ]
    )

    writer.writeheader()

    writer.writerows(
        results
    )


print()
print("=" * 75)
print(
    f"Saved: {output_file}"
)
print("=" * 75)
print()
