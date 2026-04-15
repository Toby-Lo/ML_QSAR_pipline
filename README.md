# ML Simple QSAR Pipeline

This repository contains a compact QSAR and virtual screening workflow for binary activity prediction.

## Features

- Binary QSAR modeling with classic machine learning methods
- 2048-bit Morgan fingerprints plus RDKit descriptors
- Scaffold-aware train/test splitting and grouped cross-validation
- Optional hyperparameter tuning and threshold optimization
- Probability calibration with sigmoid or isotonic methods
- Model robustness checks by Y-scrambling
- Applicability domain (AD) analysis with similarity and distance-based methods
- SHAP-based model interpretation
- Large-scale virtual screening with Parquet streaming
- AD-aware ranking for post-screening prioritization

## Main Workflow

1. Clean and curate the activity dataset
2. Train QSAR models
3. Optionally run calibration, robustness, AD, and interpretation
4. Prepare the screening library
5. Generate screening features
6. Run virtual screening inference
7. Re-rank hits with AD-aware filtering

## Environment

Create the conda environment from the provided file:

```bash
conda env create -f env/env.yaml
conda activate qsar_ml_env
```

## Quick Start

Train the main QSAR models:

```bash
python scripts/step10_qsar_ml.py --config config/nsd2_ml.yaml
```

Run probability calibration:

```bash
python scripts/step20_calibration.py \
  --run-dir models_out/qsar_ml_YYYYMMDD_HHMMSS \
  --input data/NSD2/nsd2_final_dataset_feature_fingerprint.csv \
  --methods both \
  --calibration-source dev
```

Run applicability domain analysis for one model:

```bash
python scripts/step22_applicability_domain.py \
  --run-dir models_out/qsar_ml_YYYYMMDD_HHMMSS \
  --split-seed 42 \
  --model SVC
```

Run virtual screening inference:

```bash
python scripts/step33_vs_inference.py \
  --model_dir models_out/qsar_ml_YYYYMMDD_HHMMSS \
  --model_name SVC \
  --seed 42 \
  --input data/database/zinc_features.parquet \
  --ad_integration
```

Re-rank screening hits:

```bash
python scripts/step34_ad_aware_filter.py \
  --input-file models_out/qsar_ml_YYYYMMDD_HHMMSS/virtual_screening/zinc_predictions_TIMESTAMP.parquet \
  --top-n 10000 \
  --ad-power 2.0 \
  --ad-threshold 0.3
```

## Key Files

- `config/nsd2_ml.yaml`: main training configuration
- `config/smoke_test.yaml`: small test configuration
- `scripts/step10_qsar_ml.py`: main QSAR training pipeline
- `scripts/step33_vs_inference.py`: main screening inference pipeline
- `methodology/Methodology_Draft.md`: manuscript-style methods draft

## Scripts

### Data Preparation

| Script | What it does | Simple usage |
|---|---|---|
| `scripts/step01_data_cleaning.py` | Merges raw NSD2 activity/compound tables, cleans SMILES, removes duplicates, creates labels, and exports the curated dataset. This is a notebook-style script. | Run interactively in VS Code/Jupyter after checking the input paths. |
| `scripts/step02_data_analysis_cluster.py` | Optional exploratory analysis of the curated dataset using PCA, t-SNE, clustering, and enrichment plots. | `python scripts/step02_data_analysis_cluster.py` |

### Training

| Script | What it does | Simple usage |
|---|---|---|
| `scripts/step10_qsar_ml.py` | Main training script. Builds features, splits data, runs cross-validation, trains models, selects thresholds, evaluates external test performance, and saves artifacts. | `python scripts/step10_qsar_ml.py --config config/nsd2_ml.yaml` |
| `scripts/step11_training_summary.py` | Aggregates per-seed external metrics and helps identify representative seeds. | `python scripts/step11_training_summary.py --run-dir models_out/qsar_ml_YYYYMMDD_HHMMSS` |

### Validation and Analysis

| Script | What it does | Simple usage |
|---|---|---|
| `scripts/step20_calibration.py` | Calibrates model probabilities from a step10 run using sigmoid and/or isotonic calibration. | `python scripts/step20_calibration.py --run-dir models_out/qsar_ml_YYYYMMDD_HHMMSS --input data/NSD2/nsd2_final_dataset_feature_fingerprint.csv --methods both --calibration-source dev` |
| `scripts/step21_model_robustness.py` | Runs Y-scrambling to test whether model performance is better than label-randomized baselines. | `python scripts/step21_model_robustness.py --run-dir models_out/qsar_ml_YYYYMMDD_HHMMSS --split-seed 42 --models SVC --n-permutations 200 --input data/NSD2/nsd2_final_dataset_feature_fingerprint.csv` |
| `scripts/step22_applicability_domain.py` | Computes applicability domain scores and in/out-of-domain flags for external predictions. | `python scripts/step22_applicability_domain.py --run-dir models_out/qsar_ml_YYYYMMDD_HHMMSS --split-seed 42 --model SVC` |

### Interpretation

| Script | What it does | Simple usage |
|---|---|---|
| `scripts/step23_interpretations_tree.py` | SHAP analysis for tree models such as RFC, ETC, and XGBC. | `python scripts/step23_interpretations_tree.py --run-dir models_out/qsar_ml_YYYYMMDD_HHMMSS --split-seed 42 --models RFC,ETC,XGBC` |
| `scripts/step24_interpretations_linear.py` | SHAP analysis for linear/SVC-style models using linear or kernel explainers depending on the model. | `python scripts/step24_interpretations_linear.py --run-dir models_out/qsar_ml_YYYYMMDD_HHMMSS --split-seed 42` |
| `scripts/step25_interpretations_kernel.py` | Kernel SHAP analysis for slower nonlinear models such as MLP and SVC. | `python scripts/step25_interpretations_kernel.py --run-dir models_out/qsar_ml_YYYYMMDD_HHMMSS --split-seed 42 --models MLP,SVC` |

### Virtual Screening

| Script | What it does | Simple usage |
|---|---|---|
| `scripts/step30_vs_preparation.py` | First-pass Parquet filtering of a large screening library using basic RDKit and PAINS checks. | `python scripts/step30_vs_preparation.py` |
| `scripts/step31_vs_druglike_filter.py` | Applies a stricter drug-likeness filter to the screening library. | `python scripts/step31_vs_druglike_filter.py` |
| `scripts/step32_vs_features.py` | Builds Morgan fingerprints and RDKit descriptor columns for the filtered screening set. | `python scripts/step32_vs_features.py --config config/nsd2_ml.yaml --input data/database/zinc_druglike.parquet --output data/database/zinc_features.parquet` |
| `scripts/step33_vs_inference.py` | Runs batch QSAR inference on the screening feature table and can attach AD scores during prediction. | `python scripts/step33_vs_inference.py --model_dir models_out/qsar_ml_YYYYMMDD_HHMMSS --model_name SVC --seed 42 --input data/database/zinc_features.parquet --ad_integration` |
| `scripts/step34_ad_aware_filter.py` | Filters and re-ranks screening predictions using AD-aware scoring. | `python scripts/step34_ad_aware_filter.py --input-file path/to/zinc_predictions.parquet --top-n 10000 --ad-power 2.0 --ad-threshold 0.3` |

### Plotting

| Script | What it does | Simple usage |
|---|---|---|
| `scripts/step40_plot_performance.py` | Plots ROC/PR curves and metric boxplots from a training run. | `python scripts/step40_plot_performance.py --base-dir models_out/qsar_ml_YYYYMMDD_HHMMSS --include-external --include-cv` |
| `scripts/step41_threshold_analysis.py` | Plots threshold-dependent ROC, PR, F1, and MCC curves for selected models and seeds. | `python scripts/step41_threshold_analysis.py --base-dir models_out/qsar_ml_YYYYMMDD_HHMMSS --model SVC --seed 42` |

## Input and Output Notes

- Training data are expected as CSV or Parquet tables with SMILES, labels, and optionally precomputed feature columns.
- Virtual screening uses Parquet files and assumes at least `zinc_id` and `smiles`.
- Step10 writes all model artifacts under `models_out/qsar_ml_YYYYMMDD_HHMMSS/`.
- Later scripts read from that run directory instead of rebuilding models from scratch.

## Recommended Order

For model development:

```bash
step01 -> step10 -> step20 -> step21 -> step22 -> step23/24/25 -> step40/41
```

For virtual screening:

```bash
step30 -> step31 -> step32 -> step33 -> step34
```
