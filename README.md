# Robust QSAR via Nested Scaffold-Grouped Cross-Validation: An In Silico Screening Pipeline

This repository provides an end-to-end, ML-only QSAR + virtual screening workflow for **binary activity prediction**, with a strong emphasis on **AD-aware (Applicability Domain-aware) ranking** and **ADMET-aware post-screening prioritization**.

## Features

- **Scaffold-aware evaluation**: Bemis–Murcko scaffold grouping for splits and CV (reduces scaffold leakage).
- **Nested-CV-style training**: outer dev/external split + inner (scaffold-grouped) CV for tuning/threshold selection.
- **Feature stack**: 2048-bit Morgan fingerprints + aligned RDKit descriptors.
- **Modeling**: classic ML baselines (LR/SVC/RF/ET/MLP/XGBoost), optional HPO, threshold optimization.
- **Calibration**: sigmoid / isotonic probability calibration (post-hoc).
- **Robustness**: Y-scrambling sanity checks.
- **Applicability Domain (AD)**: leverage (PCA) + similarity (Tanimoto/cosine) + density; optional weight learning.
- **Interpretability**: SHAP pipelines for tree / linear / kernel explainers.
- **Virtual screening at scale**: Parquet streaming I/O and batch inference.
- **AD-aware + ADMET-aware ranking**: multiple “final score” layers for practical prioritization.

## Main Workflow

1. Clean and curate the activity dataset
2. Train QSAR models
3. Optionally run calibration, robustness, AD, and interpretation
4. Prepare the screening library
5. Generate screening features
6. Run virtual screening inference
7. Re-rank hits with AD-aware filtering
8. (Optional) Merge **ADMETlab 3.0** predictions and compute an ADMET-aware final score

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

Summarize external-test metrics across seeds:

```bash
python scripts/step11_training_summary.py --run-dir models_out/qsar_ml_YYYYMMDD_HHMMSS
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

Optionally add ADMET scoring (requires an ADMETlab 3.0 batch result CSV):

```bash
python scripts/step35_admet_score.py \
  --input-parquet models_out/qsar_ml_YYYYMMDD_HHMMSS/virtual_screening/.../top_hits.parquet \
  --admetlab-file models_out/qsar_ml_YYYYMMDD_HHMMSS/virtual_screening/admet/ADMETlab3_result.csv \
  --admet-smiles-col smiles \
  --mode default \
  --no-include-bbb \
  --include-dili
```

## Key Files

- `config/nsd2_ml.yaml`: main training configuration
- `config/smoke_test.yaml`: small test configuration
- `scripts/step10_qsar_ml.py`: main QSAR training pipeline
- `scripts/step33_vs_inference.py`: main screening inference pipeline
- `methodology/Methodology_Draft.md`: manuscript-style methods draft

## Scores and Ranking (Full “Final Score” Definitions)

This project intentionally exposes multiple scoring layers so you can choose a ranking strategy that matches your risk tolerance and downstream cost (e.g., docking).

### 1) AD components (step22 / step33)

The AD score is built from similarity + density:

- `Sim_Score = w1*Tanimoto + w2*Cosine` where `w1 + w2 = 1`
- `AD_Score = w3*Sim_Score + w4*Density_Score` where `w3 + w4 = 1`
- Optional power transform: `AD_Score_Powered = (AD_Score)^k` (controlled by `--ad-score-power`)

### 2) QSAR × AD fused scores (step22)

Given calibrated (or raw) QSAR probability `qsar_prob` (a.k.a. `y_prob`) and `AD_Score_Powered`:

- `Final_Score = clip(qsar_prob, eps, 1-eps) * AD_Score_Powered`
- `Final_Score_Shrunk`: AD-aware “probability shrinkage” score (method controlled by `--logit-shrinkage-method`; falls back to the same formula as `Final_Score` if shrinkage is unavailable)

Step22 exports these columns (names in outputs):
- `Sim_Score`, `Density_Score`, `AD_Score`
- `Final_Score`, `Final_Score_Shrunk` (also aliased as `Final_Score_Logit`)

### 3) AD-aware screening re-ranking (step34)

Step34 is a fast post-processing layer over a parquet that already contains `qsar_prob` and `ad_score`:

- `qsar_ad_rank_score_raw = qsar_prob * (ad_score ^ ad_power)` (when `--use-shrinkage`)
- `qsar_ad_rank_score = minmax_01(qsar_ad_rank_score_raw)` (pretty normalized to `[0, 1]` for ranking/plotting)

It also applies an **AD threshold** filter: keep rows with `ad_score >= ad_threshold`.

### 4) ADMET-aware final score (step35)

Step35 merges ADMETlab 3.0 endpoint predictions and computes:

- `admet_score` in `[0, 1]` from sub-scores (Absorption / Distribution / Metabolism / Toxicity; optional Excretion)
- `final_score_raw = 0.5*qsar_prob + 0.2*ad_score + 0.3*admet_score`
- `final_score = minmax_01(final_score_raw)` (normalized to `[0, 1]`)

## Scripts

Naming convention:
- `stepXX_*.py`: `XX` indicates pipeline stage order.
- `step01-02`: dataset curation and exploratory analysis.
- `step10-11`: core QSAR model training and cross-seed summary.
- `step20-25`: calibration, robustness, AD analysis, and interpretation.
- `step30-35`: virtual screening, AD-aware ranking, and ADMET-aware prioritization.
- `step40-42`: figure generation and screening-result visualization.

### Data Preparation

#### `scripts/step01_data_cleaning.py`

- What it does: notebook-style data cleaning/curation (merge raw tables, SMILES cleaning, de-duplication, label creation, exports).
- Simple usage: run interactively (contains notebook cells / magics).

#### `scripts/step02_data_analysis_cluster.py`

- What it does: optional EDA (Morgan FP, scaling + PCA, t-SNE, clustering, and figure export).
- Simple usage: `python scripts/step02_data_analysis_cluster.py`

### Training

#### `scripts/step10_qsar_ml.py`

- What it does: main QSAR training pipeline (feature building, scaffold-aware splitting, scaffold-grouped CV, optional HPO, OOF threshold selection, external-test evaluation, artifact export including SHAP-ready bundles).
- Simple usage: `python scripts/step10_qsar_ml.py --config config/nsd2_ml.yaml`

#### `scripts/step11_training_summary.py`

- What it does: seed-level aggregator (reads each `split_seed_*` external summary, reports mean/std across seeds, helps pick a representative seed).
- Simple usage: `python scripts/step11_training_summary.py --run-dir models_out/qsar_ml_YYYYMMDD_HHMMSS`

### Validation and Analysis

#### `scripts/step20_calibration.py`

- What it does: post-hoc probability calibration (sigmoid / isotonic) for step10 models, with diagnostics and exported calibrated artifacts.
- Simple usage: `python scripts/step20_calibration.py --run-dir models_out/qsar_ml_YYYYMMDD_HHMMSS --input data/NSD2/nsd2_final_dataset_feature_fingerprint.csv --methods both --calibration-source dev`

#### `scripts/step21_model_robustness.py`

- What it does: Y-scrambling robustness test (real labels vs permuted-label baselines; plots supported).
- Simple usage: `python scripts/step21_model_robustness.py --run-dir models_out/qsar_ml_YYYYMMDD_HHMMSS --split-seed 42 --models SVC --n-permutations 200 --input data/NSD2/nsd2_final_dataset_feature_fingerprint.csv`

#### `scripts/step22_applicability_domain.py`

- What it does: AD analysis for step10 outputs (leverage/PCA/Williams + similarity + density; optional weight learning; optional calibration comparisons); exports `AD_Score` and fused “Final_Score” variants.
- Simple usage: `python scripts/step22_applicability_domain.py --run-dir models_out/qsar_ml_YYYYMMDD_HHMMSS --split-seed 42 --model SVC`

### Interpretation

#### `scripts/step23_interpretations_tree.py`

- What it does: SHAP interpretation for tree models (RFC/ETC/XGBC) using step10-exported SHAP bundles.
- Simple usage: `python scripts/step23_interpretations_tree.py --run-dir models_out/qsar_ml_YYYYMMDD_HHMMSS --split-seed 42 --models RFC,ETC,XGBC`

#### `scripts/step24_interpretations_linear.py`

- What it does: SHAP interpretation for LR/SVC; switches explainer (linear vs kernel) depending on the model.
- Simple usage: `python scripts/step24_interpretations_linear.py --run-dir models_out/qsar_ml_YYYYMMDD_HHMMSS --split-seed 42`

#### `scripts/step25_interpretations_kernel.py`

- What it does: Kernel SHAP for slower nonlinear models (e.g., MLP, SVC forced kernel mode), with built-in downsampling for practicality.
- Simple usage: `python scripts/step25_interpretations_kernel.py --run-dir models_out/qsar_ml_YYYYMMDD_HHMMSS --split-seed 42 --models MLP,SVC`

### Virtual Screening

#### `scripts/step30_vs_preparation.py`

- What it does: stage 1 library filtering (streaming Parquet): fast heuristics + RDKit physchem rules + PAINS removal.
- Simple usage: `python scripts/step30_vs_preparation.py`

#### `scripts/step31_vs_druglike_filter.py`

- What it does: stage 2 stricter filtering: allowed atoms, tighter physchem ranges, and QED cutoff; streaming Parquet I/O.
- Simple usage: `python scripts/step31_vs_druglike_filter.py`

#### `scripts/step32_vs_features.py`

- What it does: stage 3 feature store aligned to training: Morgan bits (`morgan_0..morgan_2047`) + RDKit descriptors using the same descriptor list as step10.
- Simple usage: `python scripts/step32_vs_features.py --config config/nsd2_ml.yaml --input data/database/zinc_druglike.parquet --output data/database/zinc_features.parquet`

#### `scripts/step33_vs_inference.py`

- What it does: production inference over Parquet feature tables with strict feature schema alignment; optional real-time AD integration using step22 artifacts.
- Simple usage: `python scripts/step33_vs_inference.py --model_dir models_out/qsar_ml_YYYYMMDD_HHMMSS --model_name SVC --seed 42 --input data/database/zinc_features.parquet --ad_integration`

#### `scripts/step34_ad_aware_filter.py`

- What it does: fast AD-aware re-ranking: apply AD thresholding, compute `qsar_ad_rank_score_raw`, and export top-N/top-% hits.
- Simple usage: `python scripts/step34_ad_aware_filter.py --input-file path/to/zinc_predictions.parquet --top-n 10000 --ad-power 2.0 --ad-threshold 0.3`

#### `scripts/step35_admet_score.py`

- What it does: ADMET scoring layer (no hard filtering): merge ADMETlab 3.0 batch CSV, compute `admet_score`, and produce `final_score_raw/final_score` for final prioritization.
- Simple usage: `python scripts/step35_admet_score.py --input-parquet path/to/top_hits.parquet --admetlab-file path/to/ADMETlab3_result.csv --admet-smiles-col smiles`

### Plotting

#### `scripts/step40_plot_performance.py`

- What it does: plots ROC/PR curves and metric boxplots from a training run.
- Simple usage: `python scripts/step40_plot_performance.py --base-dir models_out/qsar_ml_YYYYMMDD_HHMMSS --include-external --include-cv`

#### `scripts/step41_threshold_analysis.py`

- What it does: plots threshold-dependent ROC, PR, F1, and MCC curves for selected models/seeds.
- Simple usage: `python scripts/step41_threshold_analysis.py --base-dir models_out/qsar_ml_YYYYMMDD_HHMMSS --model SVC --seed 42`

#### `scripts/step42_vs_visualization.py`

- What it does: builds publication-style virtual screening funnel figures (stage attrition + score distributions) and exports stage-count/summary tables.
- Simple usage: `python scripts/step42_vs_visualization.py --run-dir models_out/qsar_ml_YYYYMMDD_HHMMSS`

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
step30 -> step31 -> step32 -> step33 -> step34 -> (optional) step35 -> step42
```
