# DCT-Enhanced Lithography Hotspot Detection

**Can frequency-domain information obtained using the Discrete Cosine Transform (DCT) improve lithography hotspot detection compared with conventional spatial-domain representations?**

A controlled empirical study on the ICCAD-12 lithography hotspot benchmark using lightweight CNN-based classifiers.

## Authors

- **Palempalli Lakshmi Narasimha**
- **A. Sanjay**
- **Anish Anil Shende**

**Vellore Institute of Technology, Chennai, India**

## Overview

This project investigates whether transforming lithography layout images into the frequency domain using the **Discrete Cosine Transform (DCT)** provides useful information for hotspot classification.

Four representations are compared under a common lightweight CNN framework:

1. **R0 — Spatial Baseline**
2. **R1 — Full DCT**
3. **R2 — Multi-band DCT**
4. **R3 — Spatial + Multi-band DCT Fusion**

The study evaluates all five ICCAD-12 benchmarks (B1–B5), using **Balanced Accuracy (BA)** as the primary metric because of severe class imbalance. Additional analyses include benchmark-wise metrics, confusion matrices, B3 frequency-band ablation, feature-level frequency analysis, and computational cost.

### Main Result

| Representation | Average Balanced Accuracy |
|---|---:|
| **R0 — Spatial Baseline** | **92.46%** |
| R1 — Full DCT | 51.54% |
| R2 — Multi-band DCT | 64.00% |
| R3 — Spatial + Multi-band DCT | 64.61% |

The experiments show that DCT representations contain discriminative information, but the tested frequency-domain representations **do not provide a robust improvement over the spatial representation under the evaluated setup**.

## Research Question

> **Can frequency-domain information obtained using Discrete Cosine Transform (DCT) improve hotspot detection compared with conventional spatial-domain representations?**

## Experimental Configurations

### R0 — Spatial Baseline

- Input: resized spatial grayscale layout image
- Resolution: `224 × 224`
- Grayscale image replicated to 3 channels
- CNN parameters: **12,873**
- Optimizer: Nadam
- Epochs: 10
- Batch size: 32

### R1 — Full DCT

```text
Grayscale image
      ↓
Resize to 224 × 224
      ↓
Normalize /255
      ↓
2-D orthonormal DCT
      ↓
Absolute magnitude
      ↓
log1p transformation
      ↓
Per-image min-max normalization
      ↓
Channel replication
      ↓
Lightweight CNN
```

- Full DCT spectrum
- Single frequency representation replicated to 3 channels
- CNN parameters: **12,873**
- Optimizer: Nadam
- Epochs: 10
- Batch size: 32

### R2 — Multi-band DCT

| Band | Definition | Number of coefficients |
|---|---|---:|
| Low | `r ≤ 0.15` | 1,806 |
| Mid | `0.15 < r ≤ 0.40` | 10,822 |
| High | `r > 0.40` | 37,548 |
| **Total** | | **50,176** |

The three bands form the three CNN input channels. The CNN contains **12,873 parameters**.

### R3 — Spatial + Multi-band DCT Fusion

This configuration combines a spatial branch and a multi-band DCT branch through feature fusion. The model is parameter-matched to approximately **12,873 parameters**.

## Dataset

The experiments use the **ICCAD-12 lithography hotspot detection benchmark**:

- B1
- B2
- B3
- B4
- B5

### Dataset Statistics

| Benchmark | Train HS | Train NHS | Test HS | Test NHS |
|---|---:|---:|---:|---:|
| B1 | 99 | 340 | 226 | 4,679 |
| B2 | 174 | 5,285 | 498 | 41,298 |
| B3 | 909 | 4,643 | 1,808 | 46,333 |
| B4 | 95 | 4,452 | 177 | 31,890 |
| B5 | 26 | 2,716 | 41 | 19,327 |
| **Total** | **1,303** | **17,436** | **2,750** | **143,627** |

Overall test-set hotspot prevalence: **1.88%**.

### Dataset Setup

The ICCAD-12 benchmark data is **not included in this repository**.

After obtaining the benchmark data through the appropriate source, place it in:

```text
iccad-official/
├── iccad1/
├── iccad2/
├── iccad3/
├── iccad4/
└── iccad5/
```

These directories are intentionally excluded from Git through `.gitignore`.

## Evaluation Metrics

Because of severe class imbalance, **Balanced Accuracy** is the primary metric.

Additional metrics:

- Precision
- Recall / Sensitivity
- Specificity
- F1-score
- Confusion matrices

```text
BA = (Sensitivity + Specificity) / 2
```

## Main Results

### Average Performance Across B1–B5

| Configuration | BA | Precision | Recall | Specificity | F1 |
|---|---:|---:|---:|---:|---:|
| **R0 — Spatial** | **92.46%** | 48.64% | 88.39% | 96.52% | 61.11% |
| R1 — Full DCT | 51.54% | 14.00% | 40.04% | 62.23% | 13.28% |
| R2 — Multi-band DCT | 64.00% | 23.50% | 60.35% | 67.66% | 18.98% |
| R3 — Spatial + DCT | 64.61% | 32.78% | 31.01% | 98.22% | 25.79% |

### Balanced Accuracy Differences

```text
R1 vs R0:  -40.92 percentage points
R2 vs R1:  +12.46 percentage points
R3 vs R2:  +0.61 percentage points
R3 vs R0:  -27.85 percentage points
```

### Benchmark-wise Balanced Accuracy

| Benchmark | R0 Spatial | R1 Full DCT | R2 Multi-band | R3 Fusion |
|---|---:|---:|---:|---:|
| B1 | 95.02% | 50.00% | 50.00% | 92.42% |
| B2 | 97.05% | 52.10% | 79.80% | 50.00% |
| B3 | 93.55% | 50.18% | 86.55% | 53.84% |
| B4 | 82.84% | 55.42% | 50.01% | 49.99% |
| B5 | 93.82% | 49.99% | 53.66% | 76.82% |
| **Average** | **92.46%** | **51.54%** | **64.00%** | **64.61%** |

## Frequency-Band Ablation — B3

| Representation | BA | Precision | Recall | Specificity | F1 |
|---|---:|---:|---:|---:|---:|
| Spatial | **93.55%** | 25.46% | 98.34% | 88.77% | 40.45% |
| Full DCT | 50.18% | 3.77% | 100.00% | 0.36% | 7.26% |
| Multi-band DCT | 86.55% | 13.96% | 96.24% | 76.86% | 24.39% |
| Low only | 50.00% | 0.00% | 0.00% | 100.00% | 0.00% |
| Mid only | 67.10% | 5.65% | 98.34% | 35.87% | 10.68% |
| High only | 50.60% | 3.80% | 100.00% | 1.19% | 7.32% |

The B3 ablation indicates that the frequency spectrum is not uniformly informative and that combining frequency regions is more effective than using any individual band alone under this setup.

## Feature-Level Frequency Analysis

Descriptive features include:

- Spatial mean
- Spatial variance
- Low-frequency energy
- Mid-frequency energy
- High-frequency energy
- High/low energy ratio
- Spectral centroid
- DC energy ratio
- Total DCT energy

This analysis characterizes frequency-domain information and **does not represent classifier performance**.

For an orthonormal DCT, total transform-domain energy is equivalent to spatial-domain energy under Parseval's theorem.

## Experimental Protocol

The official benchmark test partitions are kept separate from training.

For the spatial baseline and full-DCT configuration, the provided training partition was split into training and validation subsets using a fixed random seed.

For the multi-band, fusion, and frequency-ablation experiments, the provided training partition was used directly while retaining the official test partition as held-out evaluation data.

The official test data was not used for model training.

## Repository Structure

```text
dct-lithography-hotspot-detection/
│
├── README.md
├── .gitignore
├── baseline.py
├── baseline_b1.py
├── dct_baseline.py
├── dct_cached.py
├── dct_cached_matched.py
├── dct_model.py
├── dct_v2.py
├── dct_stats.py
├── r2_multiband_dct.py
├── r2_band_ablation_b3.py
├── r3_spatial_frequency_fusion.py
├── frequency_analysis.py
├── frequency_analysis_b1.py
├── inspect_dct.py
├── inspect_predictions.py
├── plot1.py
├── plot2.py
├── plot3.py
├── plot4.py
├── plot5.py
├── plot6.py
├── dct_results_B1.txt
├── dct_results_B2.txt
├── dct_results_B3.txt
├── dct_results_B4.txt
├── dct_results_B5.txt
├── frequency_results_B1.csv
├── frequency_results_B2.csv
├── frequency_results_B3.csv
├── frequency_results_B4.csv
├── frequency_results_B5.csv
├── Architecture.png
├── figure1_iccad_class_imbalance.png
├── figure2_balanced_accuracy.png
├── figure3_average_metrics.png
├── figure4_b3_frequency_ablation.png
├── figure5_b3_confusion_matrices.png
└── figure6_frequency_feature_auc.png
```

The ICCAD-12 dataset directories and local Python environment are intentionally excluded from version control.

## Environment Setup

Python 3.11 is recommended.

```bash
python3.11 -m venv litho_env
source litho_env/bin/activate
pip install numpy pandas scipy scikit-learn matplotlib
pip install tensorflow
```

The local `litho_env/` directory should not be committed to GitHub.

## Running the Experiments

After placing the ICCAD-12 benchmark directories in the project root:

### Spatial baseline

```bash
python baseline.py
```

### Full DCT

```bash
python dct_baseline.py
```

### Multi-band DCT

```bash
python r2_multiband_dct.py
```

### B3 frequency-band ablation

```bash
python r2_band_ablation_b3.py
```

### Spatial + frequency fusion

```bash
python r3_spatial_frequency_fusion.py
```

### Frequency feature analysis

```bash
python frequency_analysis.py
```

Exact dataset paths and runtime parameters may need to be adjusted according to the local ICCAD-12 dataset organization.

## Reproducing the Figures

```bash
python plot1.py
python plot2.py
python plot3.py
python plot4.py
python plot5.py
python plot6.py
```

## Reproducibility Notes

1. Obtain the ICCAD-12 benchmark datasets through the appropriate source.
2. Place B1–B5 in the expected directory structure.
3. Create the Python environment.
4. Install the required dependencies.
5. Run the baseline and DCT configurations.
6. Run the B3 frequency-band ablation.
7. Generate the frequency-level statistics.
8. Generate the plots from the resulting outputs.
9. Compare the results against the reported benchmark metrics.

The official test partition should remain isolated from model training.

## Key Observations

1. **Spatial representation remains highly competitive:** the spatial baseline achieves **92.46% average Balanced Accuracy**.
2. **Full-spectrum DCT is not sufficient by itself:** full DCT achieves **51.54% average Balanced Accuracy**.
3. **Frequency-band decomposition improves DCT performance:** multi-band DCT reaches **64.00%**.
4. **Spatial-frequency fusion does not automatically provide an improvement:** R3 reaches **64.61%**, below R0.
5. **Frequency information is not uniformly distributed:** B3 ablation shows different behavior across frequency regions.

## Limitations

- The experiments use a lightweight CNN rather than a large modern vision architecture.
- The frequency representation uses a fixed 2-D DCT and radial band partitioning.
- Validation protocols differ between some configurations because of the experimental workflow.
- The study evaluates the five ICCAD-12 benchmarks and does not establish generalization to unrelated lithography datasets.
- Learnable frequency transforms and more sophisticated spatial-frequency attention mechanisms are not investigated.
- The study does not claim that DCT is inherently ineffective; rather, the evaluated DCT representations did not provide robust improvement under the tested setup.

## Conclusion

This project presents a controlled empirical investigation of frequency-domain representations for lithography hotspot detection.

Across ICCAD-12 B1–B5, the conventional spatial representation achieved **92.46%** average Balanced Accuracy. Full-spectrum DCT achieved **51.54%**, multi-band DCT achieved **64.00%**, and spatial-frequency fusion achieved **64.61%**.

The results indicate that **frequency-domain information contains useful discriminative structure**, but the particular DCT representations evaluated in this study do not provide a robust improvement over the conventional spatial representation.

Future work should investigate more adaptive ways of learning and integrating frequency-domain information rather than relying solely on fixed DCT transformations and manually defined frequency bands.

## Citation

```bibtex
@misc{dct_lithography_hotspot_detection_2026,
  title        = {DCT-Enhanced Lithography Hotspot Detection},
  author       = {Palempalli Lakshmi Narasimha and
                  A. Sanjay and
                  Anish Anil Shende},
  year         = {2026},
  institution  = {Vellore Institute of Technology, Chennai, India},
  note         = {Controlled empirical study of spatial and DCT-based
                  representations for lithography hotspot detection}
}
```

## Related Work

A particularly relevant prior study is:

> H. Yang, Y. Lin, B. Yu, and E. F. Y. Young, “Lithography Hotspot Detection: From Shallow to Deep Learning,” IEEE International System-on-Chip Conference (SOCC), 2017.

The present work should therefore be understood as a **controlled empirical comparison and analysis of DCT representations**, rather than as the first application of DCT to lithography hotspot detection.

## License

This repository contains research code and experimental artifacts.

The ICCAD-12 benchmark data is **not distributed with this repository**. Users are responsible for obtaining and using the benchmark data in accordance with its applicable terms and conditions.

## Acknowledgment

This work was conducted as part of the **AI and Machine Learning for IC** coursework at **Vellore Institute of Technology, Chennai, India**.
