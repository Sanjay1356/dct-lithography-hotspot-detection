import os
import csv
import json
import numpy as np

from PIL import Image
from scipy.fft import dct
from scipy.stats import mannwhitneyu


# ============================================================
# CONFIG
# ============================================================

ROOT = os.path.expanduser(
    "~/Downloads/iccad-official/iccad1"
)

TRAIN_DIR = os.path.join(
    ROOT,
    "train"
)

IMG_SIZE = (224, 224)

# Radial frequency bands.
#
# r_norm = radial frequency / maximum possible radial frequency
#
# LOW  : 0.00 - 0.15
# MID  : 0.15 - 0.40
# HIGH : 0.40 - 1.00
#
# These boundaries are fixed BEFORE looking at class results.

LOW_LIMIT = 0.15
MID_LIMIT = 0.40


# ============================================================
# FILE COLLECTION
# ============================================================

def get_files(directory):

    return sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if (
            os.path.isfile(os.path.join(directory, f))
            and f.lower().endswith(".png")
        )
    ])


hs_files = get_files(
    os.path.join(
        TRAIN_DIR,
        "train_hs"
    )
)

nhs_files = get_files(
    os.path.join(
        TRAIN_DIR,
        "train_nhs"
    )
)

print("\n" + "=" * 70)
print("DCT FREQUENCY-DOMAIN ANALYSIS — B1")
print("=" * 70)

print(
    f"HS samples  : {len(hs_files)}"
)

print(
    f"NHS samples : {len(nhs_files)}"
)

print(
    f"Total       : {len(hs_files) + len(nhs_files)}"
)

print("=" * 70)


# ============================================================
# RADIAL FREQUENCY MASKS
# ============================================================

N = IMG_SIZE[0]

u = np.arange(N)
v = np.arange(N)

U, V = np.meshgrid(
    u,
    v,
    indexing="ij"
)

# Distance from DC coefficient (0,0)
radius = np.sqrt(
    U.astype(np.float32) ** 2 +
    V.astype(np.float32) ** 2
)

max_radius = np.sqrt(
    2 * (N - 1) ** 2
)

r_norm = radius / max_radius


LOW_MASK = (
    r_norm <= LOW_LIMIT
)

MID_MASK = (
    (r_norm > LOW_LIMIT) &
    (r_norm <= MID_LIMIT)
)

HIGH_MASK = (
    r_norm > MID_LIMIT
)


print("\nFrequency bands:")
print(
    f"LOW  : r <= {LOW_LIMIT}"
)

print(
    f"MID  : {LOW_LIMIT} < r <= {MID_LIMIT}"
)

print(
    f"HIGH : r > {MID_LIMIT}"
)


# ============================================================
# DCT FEATURE EXTRACTION
# ============================================================

def extract_features(path):

    image = Image.open(path).convert("L")

    image = image.resize(
        IMG_SIZE,
        Image.Resampling.BILINEAR
    )

    image = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    # TRUE 2-D orthonormal DCT
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

    # DCT energy
    energy = coeff ** 2

    total_energy = np.sum(
        energy
    )

    low_energy = np.sum(
        energy[LOW_MASK]
    )

    mid_energy = np.sum(
        energy[MID_MASK]
    )

    high_energy = np.sum(
        energy[HIGH_MASK]
    )

    # Avoid divide-by-zero
    if total_energy > 0:

        low_ratio = (
            low_energy /
            total_energy
        )

        mid_ratio = (
            mid_energy /
            total_energy
        )

        high_ratio = (
            high_energy /
            total_energy
        )

    else:

        low_ratio = 0.0
        mid_ratio = 0.0
        high_ratio = 0.0

    # High/low ratio
    high_low_ratio = (
        high_energy /
        (low_energy + 1e-12)
    )

    # Spectral centroid
    weighted_radius = np.sum(
        r_norm * energy
    )

    spectral_centroid = (
        weighted_radius /
        (total_energy + 1e-12)
    )

    # DC energy proportion
    dc_energy = energy[0, 0]

    dc_ratio = (
        dc_energy /
        (total_energy + 1e-12)
    )

    return {
        "low_energy_ratio":
            float(low_ratio),

        "mid_energy_ratio":
            float(mid_ratio),

        "high_energy_ratio":
            float(high_ratio),

        "high_low_ratio":
            float(high_low_ratio),

        "spectral_centroid":
            float(spectral_centroid),

        "dc_energy_ratio":
            float(dc_ratio),

        "total_energy":
            float(total_energy)
    }


# ============================================================
# PROCESS DATASET
# ============================================================

feature_names = [
    "low_energy_ratio",
    "mid_energy_ratio",
    "high_energy_ratio",
    "high_low_ratio",
    "spectral_centroid",
    "dc_energy_ratio",
    "total_energy"
]

all_rows = []

print("\nProcessing HS samples...")

for i, path in enumerate(hs_files):

    features = extract_features(
        path
    )

    features["label"] = "HS"
    features["path"] = path

    all_rows.append(
        features
    )

    if (i + 1) % 25 == 0:
        print(
            f"HS: {i + 1}/{len(hs_files)}"
        )


print("\nProcessing NHS samples...")

for i, path in enumerate(nhs_files):

    features = extract_features(
        path
    )

    features["label"] = "NHS"
    features["path"] = path

    all_rows.append(
        features
    )

    if (i + 1) % 50 == 0:
        print(
            f"NHS: {i + 1}/{len(nhs_files)}"
        )


# ============================================================
# STATISTICS
# ============================================================

def cohens_d(a, b):

    a = np.asarray(a)
    b = np.asarray(b)

    na = len(a)
    nb = len(b)

    pooled_std = np.sqrt(
        (
            (na - 1) * np.var(a, ddof=1) +
            (nb - 1) * np.var(b, ddof=1)
        )
        /
        (na + nb - 2)
    )

    if pooled_std == 0:
        return 0.0

    return (
        (np.mean(a) - np.mean(b))
        /
        pooled_std
    )


def auc_from_ranks(hs, nhs):

    """
    AUC for HS being the positive class.
    Uses the Mann-Whitney U statistic.
    """

    hs = np.asarray(hs)
    nhs = np.asarray(nhs)

    u_stat, _ = mannwhitneyu(
        hs,
        nhs,
        alternative="two-sided"
    )

    auc = (
        u_stat /
        (len(hs) * len(nhs))
    )

    return float(auc)


summary = {}


print("\n")
print("=" * 90)
print("HS vs NHS FREQUENCY FEATURE ANALYSIS")
print("=" * 90)

print(
    f"{'Feature':<24}"
    f"{'HS Mean':>12}"
    f"{'NHS Mean':>12}"
    f"{'HS Std':>12}"
    f"{'NHS Std':>12}"
    f"{'Cohen d':>12}"
    f"{'AUC':>10}"
)

print("-" * 90)


for feature in feature_names:

    hs_values = np.array([
        row[feature]
        for row in all_rows
        if row["label"] == "HS"
    ])

    nhs_values = np.array([
        row[feature]
        for row in all_rows
        if row["label"] == "NHS"
    ])

    hs_mean = np.mean(
        hs_values
    )

    nhs_mean = np.mean(
        nhs_values
    )

    hs_std = np.std(
        hs_values,
        ddof=1
    )

    nhs_std = np.std(
        nhs_values,
        ddof=1
    )

    d = cohens_d(
        hs_values,
        nhs_values
    )

    auc = auc_from_ranks(
        hs_values,
        nhs_values
    )

    # Make AUC direction intuitive:
    # > 0.5 means larger feature values
    # tend to correspond to HS.
    if auc < 0.5:
        auc_display = 1.0 - auc
    else:
        auc_display = auc

    summary[feature] = {
        "HS_mean": float(hs_mean),
        "NHS_mean": float(nhs_mean),
        "HS_std": float(hs_std),
        "NHS_std": float(nhs_std),
        "cohens_d_HS_minus_NHS": float(d),
        "auc_absolute_separation": float(auc_display)
    }

    print(
        f"{feature:<24}"
        f"{hs_mean:12.6f}"
        f"{nhs_mean:12.6f}"
        f"{hs_std:12.6f}"
        f"{nhs_std:12.6f}"
        f"{d:12.4f}"
        f"{auc_display:10.4f}"
    )


print("=" * 90)


# ============================================================
# SAVE CSV
# ============================================================

output_dir = os.path.expanduser(
    "~/Downloads/litho_frequency_analysis_B1"
)

os.makedirs(
    output_dir,
    exist_ok=True
)


csv_path = os.path.join(
    output_dir,
    "B1_frequency_features.csv"
)

with open(
    csv_path,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "label",
            "path"
        ] + feature_names
    )

    writer.writeheader()

    writer.writerows(
        all_rows
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_path = os.path.join(
    output_dir,
    "B1_frequency_summary.json"
)

with open(
    summary_path,
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


print("\nSaved:")
print(csv_path)
print(summary_path)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
