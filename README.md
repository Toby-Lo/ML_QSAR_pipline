# QSAR Pipline for Binary Activity Prediction

A minimal, **classic ML-only** QSAR (binary classification) pipeline. It trains and evaluates multiple models using **RDKit Morgan fingerprints (2048 bits)** + **RDKit descriptors**, and stores all artifacts in a reproducible run folder structure.

Main training models script: `./scripts/step10_qsar_ml.py`

## What it does

- Reads an input table (CSV/Parquet) with `smiles/label/(id)` columns and computes or reuses features:
  - Reuses fingerprints if `morgan_0 ... morgan_2047` columns exist; otherwise recomputes via RDKit
  - Reuses descriptor columns if they exist; otherwise recomputes via RDKit
- Supports `scaffold / stratified / random` Dev/External splitting; runs scaffold-grouped K-fold CV on the Dev set
- Trains a selectable model set: `LR, RFC, SVC, XGBC, ETC, MLP`
- Optional hyperparameter tuning + optional OOF-based threshold locking (then evaluates on external test)
- Writes a timestamped run directory: `models_out/qsar_ml_YYYYMMDD_HHMMSS/`

## Quickstart

1) Smoke test (small data + reduced memory footprint)

```bash
python ./scripts/step10_qsar_ml.py --config config/smoke_test.yaml
```

2) NSD2 example configuration

```bash
python ./scripts/step10_qsar_ml.py --config config/nsd2_ml.yaml
```

After the run you should see something like: `models_out/qsar_ml_20260410_124055/`.

## Configuration

`config/*.yaml` files drive `step10_qsar_ml.py`. Most fields map directly to the `MLQSARConfig` dataclass inside the script.

- `config/nsd2_ml.yaml`: “full” NSD2 training config (multiple seeds, tuning, threshold locking, etc.)
- `config/smoke_test.yaml`: fast end-to-end validation (small data, fewer folds, low parallelism)

Usage:

```bash
python ./scripts/step10_qsar_ml.py --config config/nsd2_ml.yaml
```

You can also override a subset of fields via CLI (see script args), e.g. `--input --output-root --models --seeds --test-size --split-method --folds ...`.

## Scripts (purpose + usage)

### Training & evaluation

- `scripts/step10_qsar_ml.py`: main training pipeline (feature engineering + split + CV + external test + aggregation)
  - Typical runs:
    - `python scripts/step10_qsar_ml.py --config config/smoke_test.yaml`
    - `python scripts/step10_qsar_ml.py --config config/nsd2_ml.yaml`
  - Key outputs (under `models_out/qsar_ml_*/`):
    - `split_seed_*/models/.../model.joblib` (trained model)
    - `split_seed_*/models/.../scaler.joblib` (descriptor scaler for non-tree models)
    - `split_seed_*/feature_processors/fp_mask.npy` (kept fingerprint bit mask)
    - `split_seed_*/predictions/` (CV/external predictions)
    - `split_seed_*/results/` (metrics, threshold curves, threshold selection, etc.)
    - `results/` (cross-seed aggregations: `all_seed_*`)

### Probability calibration

- `scripts/step20_calibration.py`: calibrate probability outputs from step10 artifacts (sigmoid / isotonic)
  - Example:
    ```bash
    python scripts/step20_calibration.py \
      --run-dir models_out/qsar_ml_YYYYMMDD_HHMMSS \
      --input data/test_data_feature_fingerprint.csv \
      --methods both \
      --calibration-source dev
    ```
  - Outputs:
    - `split_seed_*/calibration/{MODEL}/method_{sigmoid|isotonic}/` (calibrated model + curves/metrics)
    - `figures/calibration/...` (reliability plots)

### Robustness (Y-scrambling)

- `scripts/step21_model_robustness.py`: Y-scrambling robustness check for a selected split seed, plus diagnostic plots
  - Example:
    ```bash
    python scripts/step21_model_robustness.py \
      --run-dir models_out/qsar_ml_YYYYMMDD_HHMMSS \
      --split-seed 42 \
      --models LR,RFC,SVC,XGBC,ETC,MLP \
      --n-permutations 200 \
      --input data/test_data_feature_fingerprint.csv
    ```
  - Outputs:
    - `split_seed_*/robustness/{MODEL}/y_scrambling_*.{csv,json}`
    - `figures/robustness/...` (histogram + correlation scatter)

### Plotting (performance & threshold)

- `scripts/step40_plot_performance.py`: collect prediction files from a run directory and draw ROC/PR curves + metric boxplots
  - Example:
    ```bash
    python scripts/step40_plot_performance.py --base-dir models_out/qsar_ml_YYYYMMDD_HHMMSS --include-external --include-cv
    ```
  - Outputs: defaults to `<base-dir>/figures/`

- `scripts/step41_threshold_analysis.py`: reads threshold curve data produced by step10 and draws per-seed/per-model panels
  - Example:
    ```bash
    python scripts/step41_threshold_analysis.py --base-dir models_out/qsar_ml_YYYYMMDD_HHMMSS
    ```
  - Outputs: defaults to `<base-dir>/figures/threshold_analysis/`

## Data utilities (optional)

- `scripts/step01_data_cleaning.py`: NSD2 data cleaning + pIC50 calculation + labeling + basic figures (exported from a notebook-style workflow)
  - Note: this file contains a Jupyter magic line (`%matplotlib inline`) and will fail if run with plain `python`. Run it in a notebook/interactive environment, or remove that line before CLI execution.
  - Main output: `data/NSD2/nsd2_final_dataset.csv`

- `scripts/step02_data_analysis_cluster.py`: chemical space analysis using Morgan fingerprints (PCA/t-SNE + KMeans) and scaffold stats; saves figures to `data/NSD2/cluster/`
  - Run:
    ```bash
    python scripts/step02_data_analysis_cluster.py
    ```

## Placeholders (currently empty)

These files are currently empty (0 bytes) and are reserved for future extensions:

- `scripts/step22_applicability_domain.py`
- `scripts/step23_interpretations_tree.py`
- `scripts/step30_vs_preparation.py`
- `scripts/step31_vs_inference.py`
