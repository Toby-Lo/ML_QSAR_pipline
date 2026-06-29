# Robust QSAR via Nested Scaffold-Grouped Cross-Validation

This repository contains an end-to-end QSAR and virtual screening workflow for binary activity prediction. The pipeline focuses on scaffold-aware training, post-hoc calibration, applicability-domain-aware ranking, and ADMET-based prioritization.

## What It Covers

- Data cleaning and basic exploratory analysis
- QSAR model training with scaffold-grouped cross-validation
- Calibration, robustness checks, AD analysis, and SHAP interpretation
- Virtual screening preparation, inference, and post-ranking
- ADMET scoring and weight-sensitivity analysis
- Performance and screening-funnel visualization

## Repository Structure

- `config/` - YAML configuration files for model training and screening
- `data/` - Input datasets and processed molecular feature tables
- `env/` - Conda environment files
- `scripts/` - Step-wise QSAR, calibration, AD, interpretation, and screening scripts
- `models_out/` - Model artifacts, split-specific results, calibration files, AD outputs, and screening results
- `figures/` - Generated figures and summary plots

## Input Data Requirements

The training dataset should contain standardized molecular structures and binary activity labels. At minimum, the QSAR workflow expects compound identifiers, canonical SMILES, activity labels, and the molecular features generated from the preprocessing scripts. Virtual-screening input files should contain SMILES strings and compound identifiers, such as ZINC IDs, when available.

Typical columns include:

- `compound_id`
- `smiles` or `canonical_smiles`
- `pIC50`
- `label`

Screening files are usually stored as Parquet tables for efficient batch inference.

## Reproducibility

The workflow uses scaffold-aware data partitioning and fixed random seeds to improve reproducibility. In the manuscript-associated NSD2 analysis, repeated scaffold partitions were generated across multiple split seeds, and the final representative SVC realization was based on split seed `12345`. Downstream calibration, applicability-domain analysis, Y-scrambling, SHAP interpretation, and virtual screening should be performed using the same trained model artifacts and feature schema.

## Environment

Create the conda environment with:

```bash
conda env create -f env/env.yaml
conda activate qsar_ml_env
```

## Typical Workflow

### Model Development

```bash
step01 -> step02 -> step10 -> step11 -> step20 -> step21 -> step22 -> step23/24/25 -> step40/41
```

### Virtual Screening

```bash
step30 -> step31 -> step32 -> step33 -> step34 -> step35 -> step36 -> step42
```

## Quick Start

Train QSAR models:

```bash
python scripts/step10_qsar_ml.py --config config/nsd2_ml.yaml
```

Summarize results across seeds:

```bash
python scripts/step11_training_summary.py --run-dir models_out/qsar_ml_YYYYMMDD_HHMMSS
```

Run calibration:

```bash
python scripts/step20_calibration.py \
  --run-dir models_out/qsar_ml_YYYYMMDD_HHMMSS \
  --input data/NSD2/nsd2_final_dataset_feature_fingerprint.csv \
  --methods both \
  --calibration-source dev
```

Run AD analysis:

```bash
python scripts/step22_applicability_domain.py \
  --run-dir models_out/qsar_ml_YYYYMMDD_HHMMSS \
  --split-seed 12345 \
  --model SVC
```

Run virtual screening inference:

```bash
python scripts/step33_vs_inference.py \
  --model_dir models_out/qsar_ml_YYYYMMDD_HHMMSS \
  --model_name SVC \
  --seed 12345 \
  --input data/database/zinc_features.parquet \
  --ad_integration
```

Apply AD-aware re-ranking:

```bash
python scripts/step34_ad_aware_filter.py \
  --input-file models_out/qsar_ml_YYYYMMDD_HHMMSS/virtual_screening/zinc_predictions_TIMESTAMP.parquet \
  --top-n 10000 \
  --ad-power 2.0 \
  --ad-threshold 0.3
```

Add ADMET scoring:

```bash
python scripts/step35_admet_score.py \
  --input-parquet models_out/qsar_ml_YYYYMMDD_HHMMSS/virtual_screening/.../top_hits.parquet \
  --admetlab-file models_out/qsar_ml_YYYYMMDD_HHMMSS/virtual_screening/admet/ADMETlab3_result.csv \
  --admet-smiles-col smiles \
  --mode default
```

Test score-weight sensitivity:

```bash
python scripts/step36_weight_sensitivity_test.py \
  --input-parquet models_out/qsar_ml_YYYYMMDD_HHMMSS/virtual_screening/.../admet_scored.parquet \
  --top-ns 50,100,500
```

## Script Overview

### Data Preparation

- `scripts/step01_data_cleaning.py`: clean and merge raw NSD2 activity data, remove salts, create labels, and export the curated dataset.
- `scripts/step02_data_analysis_cluster.py`: generate quick EDA, PCA/t-SNE, and clustering figures.

### Training and Validation

- `scripts/step10_qsar_ml.py`: main QSAR training pipeline with scaffold-aware splitting, CV, tuning, and artifact export.
- `scripts/step11_training_summary.py`: summarize metrics across split seeds.
- `scripts/step20_calibration.py`: apply sigmoid or isotonic calibration.
- `scripts/step21_model_robustness.py`: run Y-scrambling robustness tests.
- `scripts/step22_applicability_domain.py`: compute AD metrics and AD-aware fused scores.

### Interpretation

- `scripts/step23_interpretations_tree.py`: SHAP interpretation for tree-based models.
- `scripts/step24_interpretations_linear.py`: SHAP interpretation for linear and linear-style models.
- `scripts/step25_interpretations_kernel.py`: kernel SHAP for slower nonlinear models.

### Virtual Screening

- `scripts/step30_vs_preparation.py`: first-stage library filtering with fast heuristics and PAINS removal.
- `scripts/step31_vs_druglike_filter.py`: stricter drug-likeness filtering.
- `scripts/step32_vs_features.py`: build screening features aligned with training.
- `scripts/step33_vs_inference.py`: run batch QSAR inference on Parquet files.
- `scripts/step34_ad_aware_filter.py`: re-rank hits using AD-aware scoring and thresholds.
- `scripts/step35_admet_score.py`: merge ADMETlab predictions and compute an ADMET-aware final score.
- `scripts/step36_weight_sensitivity_test.py`: compare how different score weights affect the top-ranked compounds.

### Visualization

- `scripts/step40_plot_performance.py`: plot standard QSAR performance summaries.
- `scripts/step40_plot_performance_enhanced.py`: generate richer publication-style performance figures.
- `scripts/step41_threshold_analysis.py`: analyze threshold-dependent metrics.
- `scripts/step42_vs_visualization.py`: create screening funnel figures and summary tables.

## Main Outputs

The pipeline writes model and screening outputs under:

`models_out/qsar_ml_YYYYMMDD_HHMMSS/`

Important outputs include:

- trained model artifacts
- split-specific performance summaries
- calibration outputs
- Y-scrambling results
- applicability-domain metrics
- SHAP interpretation files
- virtual-screening prediction files
- AD-aware and ADMET-aware ranked hit tables
- screening funnel visualizations

## Notes

- Training and screening data are typically stored as CSV or Parquet tables.
- Screening input should include `smiles` and compound identifiers such as `zinc_id` when available.
- Most downstream scripts read from `models_out/qsar_ml_YYYYMMDD_HHMMSS/` and reuse artifacts from step10.


## Citation

- If you use this workflow, please cite the associated manuscript:

- Reliability-Aware and Interpretable Virtual Screening under Scaffold-Shift Conditions Prioritizes Candidate NSD2 Inhibitors.

- The full citation will be updated after publication.