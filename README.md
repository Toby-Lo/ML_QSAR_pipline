# QSAR ML Pipeline for Binary Activity Prediction

A comprehensive, **classic Machine Learning** QSAR (binary classification) pipeline designed for drug discovery and virtual screening. It trains and evaluates multiple machine learning models using **RDKit Morgan fingerprints (2048 bits)** + **RDKit molecular descriptors**, with support for advanced threshold analysis, model calibration, robustness testing, and large-scale virtual screening inference.

**Primary Training Script:** [scripts/step10_qsar_ml.py](scripts/step10_qsar_ml.py)

**Primary Inference Script:** [scripts/step33_vs_inference.py](scripts/step33_vs_inference.py)

---

## Overview

This pipeline implements a complete QSAR workflow:

1. **Feature Engineering:** 
   - Morgan fingerprints (2048 bits) with variance & correlation filtering
   - 20 RDKit molecular descriptors
   - Optional feature augmentation and missing-value imputation strategies

2. **Model Training & Cross-Validation:**
   - Flexible model selection: Logistic Regression, Random Forest, SVM, XGBoost, Extra Trees, Multi-layer Perceptron
   - Seed-based reproducibility (multiple random seeds for Dev/External splitting)
   - Scaffold-grouped K-fold cross-validation on development set
   - Optional hyperparameter tuning (Random/Grid search)

3. **Multi-Metric Threshold Analysis:**
   - Computes 5 optimal classification thresholds:
     - Max F1-score
     - Max Matthews Correlation Coefficient (MCC)
     - Max Precision
     - Max Recall
     - Youden-J statistic
   - Selection rule configurable per model
   - External test evaluation with locked threshold

4. **Model Calibration & Robustness:**
   - Probability calibration (sigmoid / isotonic)
   - Y-scrambling robustness analysis
   - Applicability domain assessment (placeholder)

5. **Virtual Screening Inference:**
   - Large-scale prediction on ZINC database or custom compounds
   - Flexible threshold selection via CLI (--threshold_metric parameter)
   - Efficient PyArrow-based I/O for massive datasets
   - Comprehensive logging with structured outputs

---

## Installation & Environment Setup

### Prerequisites
- Python 3.12.3 or compatible
- Virtual environment (conda recommended)

### Quick Setup

```bash
# 1. Create conda environment from env.yaml
conda env create -f env.yaml
conda activate qsar_ml_env

# 2. Verify installation
python -c "import rdkit; import sklearn; print('✓ All dependencies OK')"
```

### Manual Package Installation
If using pip instead of conda:

```bash
pip install -r <(cat << 'EOF'
numpy>=1.23.0
pandas>=1.5.0
scipy>=1.9.0
scikit-learn>=1.3.0
xgboost>=2.0.0
rdkit>=2023.09.0
joblib>=1.2.0
PyYAML>=6.0
pyarrow>=12.0.0
matplotlib>=3.6.0
seaborn>=0.12.0
EOF
)
```

### Environment Details
See [env.yaml](env.yaml) for complete dependency specification including:
- Core ML: NumPy, Pandas, SciPy, scikit-learn, XGBoost
- Chemistry: RDKit
- I/O: PyArrow, PyYAML, joblib
- Visualization: Matplotlib, Seaborn

---

## Quick Start

### Example 1: Smoke Test (5 min validation)

```bash
python scripts/step10_qsar_ml.py --config config/smoke_test.yaml
```

Expected output directory: `models_out/qsar_ml_YYYYMMDD_HHMMSS/`

### Example 2: Full NSD2 Training (multi-seed production run)

```bash
python scripts/step10_qsar_ml.py --config config/nsd2_ml.yaml
```

### Example 3: Virtual Screening Inference (post-training)

```bash
# Predict on ZINC compounds with F1-optimized threshold
python scripts/step33_vs_inference.py \
  --run-dir models_out/qsar_ml_{{ TIMESTAMP }} \
  --input data/zinc_compounds.parquet \
  --output-dir ./vs_results \
  --threshold_metric f1 \
  --batch-size 50000

# Or use MCC-optimized threshold
python scripts/step33_vs_inference.py \
  --run-dir models_out/qsar_ml_{{ TIMESTAMP }} \
  --input data/zinc_compounds.parquet \
  --output-dir ./vs_results \
  --threshold_metric mcc
```

---

## Configuration Files

All training configuration via YAML files in [config/](config/):

| File | Purpose | Scale |
|------|---------|-------|
| [config/smoke_test.yaml](config/smoke_test.yaml) | Fast validation & debugging | Small: ~1K compounds, 2 folds, 2 seeds |
| [config/nsd2_ml.yaml](config/nsd2_ml.yaml) | Production NSD2 QSAR model | Large: ~3.3K compounds, 5 folds, 10 seeds |

### Configuration Highlights (nsd2_ml.yaml)

```yaml
# Data
input_path: ./data/NSD2/nsd2_final_dataset_feature_fingerprint.csv
split_method: scaffold          # scaffold | stratified | random
test_size: 0.2                  # Dev/External split ratio

# Models
selected_models: ["LR", "RFC", "SVC", "XGBC", "ETC", "MLP"]

# Features
descriptor_names: [MolWt, MolLogP, NumHDonors, NumHAcceptors, ...]  # 20 descriptors
descriptor_missing:
  strategy: "zero"              # Fill NaN with 0.0 (compatible with inference)
  # Alternative: "nan_indicator" (doubles features with __isna columns)

# Thresholding (Multi-metric)
thresholding:
  selection_rule: "max_mcc"     # For external test; options: max_f1|max_mcc|max_precision|max_recall|youden_j
  curve_points: 501             # Dense threshold grid

# Hyperparameter Tuning (optional)
hyperparameter_tuning:
  enabled: true
  search_type: "grid"           # grid | random
  target_models: ["LR", "RFC", "SVC", "XGBC", "ETC", "MLP"]
  cv_folds: 3
  n_jobs: 16
```

---

## Scripts Reference

### Core Training Pipeline

#### [scripts/step10_qsar_ml.py](scripts/step10_qsar_ml.py) — Main QSAR Training
**Purpose:** End-to-end training with CV, threshold analysis, and multi-seed aggregation

**Key Features:**
- Feature engineering: Morgan fingerprints + RDKit descriptors
- Dev/External splitting (scaffold/stratified/random)
- Scaffold-grouped K-fold cross-validation
- 5-metric threshold optimization (F1, MCC, precision, recall, Youden-J)
- Optional hyperparameter tuning
- Cross-seed result aggregation

**Usage:**
```bash
python scripts/step10_qsar_ml.py --config config/nsd2_ml.yaml
```

**CLI Overrides:**
```bash
python scripts/step10_qsar_ml.py \
  --config config/nsd2_ml.yaml \
  --models "LR,RFC,SVC" \
  --seeds "42,43,44" \
  --folds 10 \
  --test-size 0.15
```

**Outputs** (under `models_out/qsar_ml_YYYYMMDD_HHMMSS/`):
- `split_seed_*/models/{model_name}/` — Trained model artifacts (model.joblib, scaler.joblib)
- `split_seed_*/feature_processors/` — Fingerprint masks, descriptor scalers
- `split_seed_*/predictions/` — Train/CV/test predictions (CSV)
- `split_seed_*/results/` — Metrics, curves, threshold analysis (CSV/JSON/PNG)
- `results/all_seed_*` — Cross-seed aggregations
- `logs/` — Runtime logs

**Key Output Files:**
- `all_seed_cv_summary.csv` — Cross-validation metrics across seeds
- `all_seed_external_summary.csv` — External test set metrics
- `all_seed_external_predictions.csv` — Full external predictions with probabilities
- `threshold_selection_summary.csv` — 5 optimal thresholds per model per seed
- `threshold_analysis/` — Plots of ROC/PR/F1-MCC curves with threshold markers

---

### Inference & Virtual Screening

#### [scripts/step33_vs_inference.py](scripts/step33_vs_inference.py) — Large-Scale Scoring
**Purpose:** Efficient virtual screening predictions on large compound libraries

**Key Features:**
- Flexible threshold selection (--threshold_metric: f1|youden|mcc|precision|recall)
- PyArrow streaming for memory-efficient large-dataset I/O
- Batch prediction with configurable batch size
- Structured logging (dual console + file output)
- Conflict detection between training & inference schemas

**Prerequisites:**
1. Trained model directory from step10 (contains `split_seed_*/models/`)
2. Input compound features (Parquet/CSV with precomputed Morgan fingerprints + descriptors)
3. Feature schema must match training (20 descriptors, no __isna features)

**Usage:**
```bash
# Predict with F1-optimized threshold
python scripts/step33_vs_inference.py \
  --run-dir models_out/qsar_ml_20260410_124055 \
  --input data/zinc_set.parquet \
  --output-dir ./predictions \
  --threshold_metric f1 \
  --batch-size 50000

# Predict with different metrics (each creates separate output column)
for metric in f1 mcc youden precision recall; do
  python scripts/step33_vs_inference.py \
    --run-dir models_out/qsar_ml_20260410_124055 \
    --input data/zinc_set.parquet \
    --output-dir ./predictions_${metric} \
    --threshold_metric ${metric}
done
```

**CLI Parameters:**
```
--run-dir PATH              Run directory from step10 (required)
--input PATH                Input Parquet/CSV (required)
--output-dir PATH           Output directory (default: ./inference_output)
--threshold_metric METRIC   Threshold selection (default: max_f1)
                            Options: f1|youden|mcc|precision|recall
--batch-size N              PyArrow batch size (default: 50000)
--n-jobs N                  Parallel jobs (default: 8)
```

**Outputs:**
- `zinc_predictions_YYYYMMDD_HHMMSS.parquet` — Predictions with columns:
  - `molecule_id`, `smiles`, `fingerprints`, `descriptors`
  - `{model}_probability` (per model: LR, RFC, SVC, etc.)
  - `{model}_prediction` (binary: 0/1 at selected threshold)
- `inference_YYYYMMDD_HHMMSS.log` — Execution log (model versions, thresholds used, timing)

---

### Post-Training Analysis

#### [scripts/step20_calibration.py](scripts/step20_calibration.py) — Probability Calibration
**Purpose:** Calibrate model probability estimates (sigmoid / isotonic regression)

**Usage:**
```bash
python scripts/step20_calibration.py \
  --run-dir models_out/qsar_ml_YYYYMMDD_HHMMSS \
  --input data/test_data_feature_fingerprint.csv \
  --methods both \
  --calibration-source dev
```

**Outputs:**
- `split_seed_*/calibration/{MODEL}/method_{sigmoid|isotonic}/`
- `figures/calibration/*.png` — Reliability plots

---

#### [scripts/step21_model_robustness.py](scripts/step21_model_robustness.py) — Y-Scrambling Analysis
**Purpose:** Validate model robustness via label shuffling permutation test

**Usage:**
```bash
python scripts/step21_model_robustness.py \
  --run-dir models_out/qsar_ml_YYYYMMDD_HHMMSS \
  --split-seed 42 \
  --models LR,RFC,SVC,XGBC,ETC,MLP \
  --n-permutations 200 \
  --input data/NSD2/nsd2_final_dataset_feature_fingerprint.csv
```

**Outputs:**
- `split_seed_42/robustness/{MODEL}/y_scrambling_*.{csv,json}`
- `figures/robustness/...` — P-value histograms, correlation plots

---

#### [scripts/step40_plot_performance.py](scripts/step40_plot_performance.py) — Performance Plots
**Purpose:** Generate ROC/PR curves and metric comparison plots

**Usage:**
```bash
python scripts/step40_plot_performance.py \
  --base-dir models_out/qsar_ml_YYYYMMDD_HHMMSS \
  --include-external --include-cv
```

**Outputs:** `models_out/qsar_ml_*/figures/`

---

#### [scripts/step41_threshold_analysis.py](scripts/step41_threshold_analysis.py) — Threshold Analysis Plots
**Purpose:** Multi-metric threshold visualization (ROC + PR + F1/MCC curves)

**Features:**
- **Panel 1:** ROC curve with Youden-J threshold point
- **Panel 2:** PR curve with Max-F1, Max-Precision, Max-Recall markers
- **Panel 3:** F1 & MCC curves with vlines for all 5 optimal thresholds

**Usage:**
```bash
python scripts/step41_threshold_analysis.py \
  --base-dir models_out/qsar_ml_YYYYMMDD_HHMMSS
```

**Outputs:** `models_out/qsar_ml_*/figures/threshold_analysis/`

---

### Data Processing Utilities

#### [scripts/step01_data_cleaning.py](scripts/step01_data_cleaning.py) — NSD2 Data Preparation
**Purpose:** Data cleaning, pIC50 calculation, activity labeling

**Note:** Contains Jupyter magic (`%matplotlib inline`) — run in notebook environment or remove that line

**Output:** `data/NSD2/nsd2_final_dataset.csv`

---

#### [scripts/step02_data_analysis_cluster.py](scripts/step02_data_analysis_cluster.py) — Chemical Space Analysis
**Purpose:** PCA/t-SNE clustering, scaffold statistics, chemical diversity assessment

**Usage:**
```bash
python scripts/step02_data_analysis_cluster.py
```

**Outputs:** `data/NSD2/cluster/`

---

#### [scripts/step30_vs_preparation.py](scripts/step30_vs_preparation.py), [scripts/step31_vs_druglike_filter.py](scripts/step31_vs_druglike_filter.py), [scripts/step32_vs_features.py](scripts/step32_vs_features.py)
**Status:** Virtual screening preparation pipeline (precompute features for ZINC)

---

## Project Structure

```
.
├── README.md                           # This file
├── env.yaml                            # Conda environment specification
│
├── config/
│   ├── nsd2_ml.yaml                   # Production NSD2 configuration
│   └── smoke_test.yaml                # Quick validation configuration
│
├── data/
│
├── scripts/
│   ├── step01_data_cleaning.py
│   ├── step02_data_analysis_cluster.py
│   ├── step10_qsar_ml.py              # Main training
│   ├── step20_calibration.py
│   ├── step21_model_robustness.py
│   ├── step22_applicability_domain.py # Placeholder
│   ├── step23_interpretations_tree.py
│   ├── step24_interpretations_linear.py
│   ├── step25_interpretations_kernel.py
│   ├── step30_vs_preparation.py
│   ├── step31_vs_druglike_filter.py
│   ├── step32_vs_features.py
│   ├── step33_vs_inference.py         # Virtual screening
│   ├── step40_plot_performance.py
│   └── step41_threshold_analysis.py
│
├── models_out/                        # Training outputs (auto-created)
│   └── qsar_ml_YYYYMMDD_HHMMSS/      # Per-run directory
│       ├── split_seed_{N}/
│       ├── results/
│       ├── figures/
│       └── logs/
└── 
```

---

## Workflow Example: Complete QSAR Pipeline

### Step 1: Prepare Data
```bash
# Optional: Clean raw data
python scripts/step01_data_cleaning.py
python scripts/step02_data_analysis_cluster.py
```

### Step 2: Train Models (with multi-metric thresholds)
```bash
python scripts/step10_qsar_ml.py --config config/nsd2_ml.yaml
# Output: models_out/qsar_ml_20260412_120000/
```

### Step 3: Evaluate Training Results
```bash
# View performance metrics
ls models_out/qsar_ml_20260412_120000/results/all_seed_*

# Plot ROC/PR curves
python scripts/step40_plot_performance.py \
  --base-dir models_out/qsar_ml_20260412_120000 \
  --include-external

# Visualize threshold analysis (5 metrics)
python scripts/step41_threshold_analysis.py \
  --base-dir models_out/qsar_ml_20260412_120000
```

### Step 4: Calibrate Models (optional)
```bash
python scripts/step20_calibration.py \
  --run-dir models_out/qsar_ml_20260412_120000 \
  --input data/NSD2/nsd2_final_dataset_feature_fingerprint.csv \
  --methods both
```

### Step 5: Robustness Check (optional)
```bash
python scripts/step21_model_robustness.py \
  --run-dir models_out/qsar_ml_20260412_120000 \
  --split-seed 42 \
  --n-permutations 200
```

### Step 6: Virtual Screening Inference
```bash
# Prepare ZINC features (step30-32)
python scripts/step30_vs_preparation.py --zinc-db zinc.parquet
python scripts/step32_vs_features.py --input zinc_prepared.parquet

# Run inference with flexible threshold selection
python scripts/step33_vs_inference.py \
  --run-dir models_out/qsar_ml_20260412_120000 \
  --input zinc_features.parquet \
  --output-dir ./zinc_predictions \
  --threshold_metric mcc \
  --batch-size 100000

# Inspect results
head zinc_predictions/zinc_predictions_*.parquet
```

---

## Key Configuration Concepts

### Multi-Metric Thresholding
The pipeline computes **5 optimal classification thresholds** during training:

1. **Max F1-Score** — Balances precision & recall equally
2. **Max MCC** — Balanced metric, especially for imbalanced data
3. **Max Precision** — Minimizes false positives (high precision)
4. **Max Recall** — Minimizes false negatives (high sensitivity)
5. **Youden-J Statistic** — Geometric mean of TPR and specificity

**Dynamic Selection:**
- Training: Lock ONE threshold for external test via `thresholding.selection_rule`
- Inference: Override at runtime via `--threshold_metric` parameter
- Analysis: All 5 thresholds saved in `threshold_selection_summary.csv`

### Descriptor Missing Value Strategy
The `descriptor_missing.strategy` controls how NaN values in RDKit descriptors are handled:

```yaml
descriptor_missing:
  strategy: "zero"        # ✓ Recommended: Fill with 0.0 (20 features)
  # OR
  strategy: "nan_indicator"  # Creates __isna indicator columns (40 features)
                            # ⚠️  WARNING: step33_vs_inference forbids __isna features
```

**Choice Impact:**
- `"zero"`: Simpler, compatible with inference, loses NaN information
- `"nan_indicator"`: Preserves information, incompatible with current inference script

---

## Troubleshooting

### Issue: `ValueError: Training features contain '__isna'`
**Cause:** Config uses `descriptor_missing.strategy: "nan_indicator"` but step33 forbids those features
**Solution:** Change to `strategy: "zero"` in config/nsd2_ml.yaml

### Issue: ModuleNotFoundError: No module named 'rdkit'
**Cause:** RDKit not installed or wrong Python environment
**Solution:**
```bash
conda activate qsar_ml_env
python -c "from rdkit import Chem; print('✓ RDKit OK')"
```

### Issue: Slow inference on large datasets
**Cause:** Batch size too small or insufficient parallelism
**Solution:**
```bash
python scripts/step33_vs_inference.py \
  --run-dir ... \
  --input zinc.parquet \
  --batch-size 100000 \  # Increase from default 50000
  --n-jobs 16            # Increase parallelism
```

---

## Citation & Acknowledgments

- **RDKit:** Landrum, G. (2016). RDKit: Open-source cheminformatics software
- **scikit-learn:** Pedregosa et al. (2011). Machine Learning in Python
- **XGBoost:** Chen, T. & Guestrin, C. (2016). XGBoost: Gradient Boosting Decision Trees
- **ZINC Database:** Sterling, T. & Irwin, J. J. (2015). ZINC 15 – Ligand Discovery for Everyone

---

## License & Contact

**Only for academic study**  
If you want to use these scripts, pleace cite...
If any problem, please contact:
