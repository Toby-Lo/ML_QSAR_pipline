"""Minimal ML-only QSAR training pipeline modeled"""

import argparse
import json
import logging
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:
    yaml = None

try:
    import joblib
except ImportError:
    import pickle

    class JoblibCompat:
        @staticmethod
        def dump(obj, path):
            with open(path, "wb") as f:
                pickle.dump(obj, f)

        @staticmethod
        def load(path):
            with open(path, "rb") as f:
                return pickle.load(f)

    joblib = JoblibCompat

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import Descriptors
import sklearn
try:
    from rdkit.Chem.rdMolDescriptors import MorganGenerator
    _MORGAN_GENERATOR_AVAILABLE = True
except ImportError:
    from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
    _MORGAN_GENERATOR_AVAILABLE = False
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
import xgboost
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             matthews_corrcoef, precision_recall_curve, precision_score, recall_score,
                             roc_curve,
                             roc_auc_score)
from sklearn.model_selection import (GridSearchCV, GroupKFold, RandomizedSearchCV,
                                     StratifiedGroupKFold, StratifiedKFold, train_test_split)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

RDKit_DESCRIPTOR_NAMES = [
    "MolWt",
    "MolLogP",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "TPSA",
    "HeavyAtomCount",
    "NumValenceElectrons",
    "NumAliphaticRings",
    "NumAromaticRings",
    "FractionCSP3",
    "RingCount",
    "LabuteASA",
    "VSA_EState1",
    "VSA_EState2",
    "SlogP_VSA1",
    "SlogP_VSA2",
    "SMR_VSA1",
    "SMR_VSA2",
    "EState_VSA1",
]

DEFAULT_MODELS = ["LR", "RFC", "SVC", "XGBC", "ETC", "MLP"]
DEFAULT_SEEDS = [42, 43, 44]

NON_TREE_MODELS = {"LR", "SVC", "MLP"}
TREE_MODELS = {"RFC", "ETC", "XGBC"}
EVAL_METRICS = ["accuracy", "precision", "recall", "f1", "mcc", "roc_auc", "pr_auc", "ef1", "ef5", "ef10", "nef10"]


def serialize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: serialize_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [serialize_json(v) for v in value]
    if isinstance(value, tuple):
        return [serialize_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return serialize_json(value.tolist())
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if np.isnan(value):
            return None
        return float(value)
    return value


def build_feature_names(mask: np.ndarray, descriptor_names: List[str]) -> Tuple[List[str], List[str]]:
    fp_indices = np.where(mask)[0]
    feature_names = [f"fp_{int(fp_idx)}" for fp_idx in fp_indices]
    feature_types = ["fp" for _ in fp_indices]
    feature_names.extend(descriptor_names)
    feature_types.extend(["descriptor" for _ in descriptor_names])
    return feature_names, feature_types


def feature_importance_dataframe(importances: np.ndarray,
                                 mask: np.ndarray,
                                 descriptor_names: List[str]) -> pd.DataFrame:
    feature_names, feature_types = build_feature_names(mask, descriptor_names)
    limit = min(len(feature_names), len(importances))
    df = pd.DataFrame({
        "feature_name": feature_names[:limit],
        "importance": np.asarray(importances[:limit], dtype=np.float64),
        "feature_type": feature_types[:limit],
    })
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def detect_existing_fingerprints(df: pd.DataFrame) -> Optional[np.ndarray]:
    candidates = []
    for col in df.columns:
        if col.startswith("morgan_"):
            suffix = col.split("morgan_", 1)[-1]
            if suffix.isdigit():
                candidates.append((int(suffix), col))
    if len(candidates) < 2048:
        return None
    sorted_cols = [col for _, col in sorted(candidates, key=lambda x: x[0])][:2048]
    return df[sorted_cols].astype(np.float32).to_numpy(dtype=np.float32)


def aggregate_fold_results(fold_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    summary: Dict[str, Dict[str, List[float]]] = {}
    for result in fold_results:
        for model_key, model_metrics in result["metrics"].items():
            val_metrics = model_metrics.get("val", {})
            summary.setdefault(model_key, {metric: [] for metric in EVAL_METRICS})
            for metric in EVAL_METRICS:
                val = val_metrics.get(metric)
                if val is None:
                    continue
                summary[model_key][metric].append(float(val))
    aggregated: Dict[str, Dict[str, Dict[str, float]]] = {}
    for model_key, metric_dict in summary.items():
        aggregated[model_key] = {}
        for metric, values in metric_dict.items():
            if not values:
                aggregated[model_key][metric] = {"mean": None, "std": None}
                continue
            aggregated[model_key][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
    return aggregated


def format_metrics_table(metrics_dict: Dict[str, Dict[str, float]], title: str = "") -> str:
    """Format metrics dictionary as an aligned table string."""
    rows = []
    if title:
        rows.append(f"\n{title}")
        rows.append("=" * 80)
    
    if not metrics_dict:
        return "\n(No metrics available)"
    
    # Get all unique metrics from all models
    all_metrics = set()
    for model_metrics in metrics_dict.values():
        all_metrics.update(model_metrics.keys())
    metric_list = sorted(list(all_metrics))
    
    # Build header
    header = "Model".ljust(10)
    for metric in metric_list:
        header += f" | {metric:>10}"
    rows.append(header)
    rows.append("-" * len(header))
    
    # Build rows for each model
    for model_key in sorted(metrics_dict.keys()):
        row = model_key.ljust(10)
        for metric in metric_list:
            val = metrics_dict[model_key].get(metric)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                row += f" | {'N/A':>10}"
            elif isinstance(val, str):
                row += f" | {val:>10}"
            else:
                row += f" | {float(val):>10.4f}"
        rows.append(row)
    
    return "\n" + "\n".join(rows)


def print_process_divider(logger: logging.Logger, stage: str, width: int = 80) -> None:
    """Print a visual divider for different processing stages."""
    divider = "=" * width
    logger.info(divider)
    logger.info(f"  {stage.upper()}  ".center(width, "="))
    logger.info(divider)


def print_feature_audit(logger: logging.Logger,
                        fp_raw: int = 2048,
                        fp_kept: Optional[int] = None,
                        descriptors_used: List[str] = None,
                        precomputed_found: bool = False) -> None:
    """Print a summary of feature engineering."""
    logger.info("=" * 80)
    logger.info("FEATURE ENGINEERING AUDIT".center(80))
    logger.info("=" * 80)
    if fp_kept is None:
        logger.info(f"Morgan Fingerprints: raw block size = {fp_raw} bits")
        logger.info("  Retained bits are selected per CV fold and on final-dev training only.")
    else:
        logger.info(f"Morgan Fingerprints: {fp_kept} / {fp_raw} bits retained after filtering")
        logger.info(f"  Retention rate: {100 * fp_kept / fp_raw:.1f}%")
    if precomputed_found:
        logger.info("  Status: Pre-computed fingerprints successfully reused")
    else:
        logger.info("  Status: Computed on-the-fly via RDKit")
    if descriptors_used:
        logger.info(f"RDKit Descriptors: {len(descriptors_used)} dimensions used")
        logger.info(f"  Names: {', '.join(descriptors_used[:5])}{'...' if len(descriptors_used) > 5 else ''}")
    logger.info("=" * 80)


def log_runtime_info(logger: logging.Logger) -> None:
    logger.info("Runtime environment:")
    logger.info(f"  Python: {sys.version.split()[0]}")
    logger.info(f"  Platform: {platform.platform()}")
    logger.info(f"  NumPy: {np.__version__}")
    logger.info(f"  pandas: {pd.__version__}")
    logger.info(f"  scikit-learn: {sklearn.__version__}")
    logger.info(f"  xgboost: {xgboost.__version__}")
    logger.info(f"  RDKit: {rdBase.rdkitVersion}")


@dataclass
class MLQSARConfig:
    input_path: Path = Path("data/qsar.csv")
    output_root: Path = Path("models_out")
    task: str = "classification"
    smiles_column: str = "smiles"
    label_column: str = "label"
    id_column: str = "id"
    seed: int = 42
    split_seeds: List[int] = field(default_factory=lambda: DEFAULT_SEEDS.copy())
    folds: int = 5
    selected_models: List[str] = field(default_factory=lambda: DEFAULT_MODELS.copy())
    descriptor_names: List[str] = field(default_factory=lambda: RDKit_DESCRIPTOR_NAMES.copy())
    split_method: str = "scaffold"
    test_size: float = 0.2
    variance_threshold: float = 0.05
    correlation_threshold: float = 0.9
    augmentation: Dict[str, Any] = field(default_factory=lambda: {
        "use_smiles_augmentation": False,
        "n_augments": 0,
        "method": "random_smiles",
        "apply_to": "train_only",
        "descriptor_mode": "reuse_original",
    })
    hyperparameter_tuning: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "search_type": "random",
        "target_models": ["LR", "RFC", "SVC", "XGBC", "ETC", "MLP"],
        "n_iter": 60,
        "cv_folds": 3,
        "scoring": "roc_auc",
        "n_jobs": -1,
    })
    thresholding: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "classification_threshold": 0.5,
        "tune_in_cv": False,
        "tune_metric": "mcc",
        "candidate_thresholds": [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7],
        "selection_rule": "max_f1",
        "curve_points": 201,
    })
    descriptor_missing: Dict[str, Any] = field(default_factory=lambda: {
        # "zero": fill missing descriptor values with 0.0 (legacy behavior)
        # "nan_indicator": keep NaN semantics via extra indicator features, then impute values with 0.0
        "strategy": "nan_indicator",
        "impute_value": 0.0,
        "indicator_suffix": "__isna",
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train simplified QSAR ML stack")
    parser.add_argument("--config", type=Path, help="Optional YAML/JSON config file")
    parser.add_argument("--input", type=Path, help="Input CSV/Parquet path")
    parser.add_argument("--output-root", type=Path, help="Root output directory (default: models_out)")
    parser.add_argument("--models", help="Comma-separated list of models")
    parser.add_argument("--seeds", help="Comma-separated seeds for splits")
    parser.add_argument("--test-size", type=float, help="External test set fraction")
    parser.add_argument("--split-method", choices=["scaffold", "stratified", "random"],
                        help="Full dataset split method")
    parser.add_argument("--folds", type=int, help="Number of folds for CV")
    parser.add_argument("--variance-threshold", type=float, help="Fingerprint variance threshold")
    parser.add_argument("--correlation-threshold", type=float, help="Fingerprint correlation threshold")
    parser.add_argument("--descriptor-names", help="Comma-separated RDKit descriptors")
    return parser.parse_args()


def load_config_file(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    if path.suffix.lower() in {".yml", ".yaml"}:
        if yaml is None:
            raise RuntimeError("yaml library is required to read YAML config files")
        with open(path) as f:
            return yaml.safe_load(f) or {}
    with open(path) as f:
        return json.load(f)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def get_scaffold(smiles: str) -> str:
    if not smiles:
        return ""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None:
            return ""
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return ""


def scaffold_split(smiles_list: List[str], y: np.ndarray, test_size: float = 0.2, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)
    scaffolds = [get_scaffold(s) for s in smiles_list]
    scaffold_dict: Dict[str, List[int]] = {}
    for idx, scaffold in enumerate(scaffolds):
        scaffold_dict.setdefault(scaffold, []).append(idx)
    scaffold_sizes = {scaffold: len(indices) for scaffold, indices in scaffold_dict.items()}
    sorted_scaffolds = sorted(scaffold_sizes.keys(), key=lambda x: scaffold_sizes[x], reverse=True)
    n_samples = len(smiles_list)
    n_test_target = int(n_samples * test_size)
    y_arr = np.asarray(y).astype(int)
    total_pos = int(np.sum(y_arr == 1))
    total_neg = int(n_samples - total_pos)
    target_pos = int(round(total_pos * test_size)) if total_pos > 0 else 0
    target_neg = int(round(total_neg * test_size)) if total_neg > 0 else 0
    min_pos_required = 1 if target_pos > 0 else 0
    min_neg_required = 1 if target_neg > 0 else 0
    best_train: Optional[List[int]] = None
    best_test: Optional[List[int]] = None
    best_score = float("inf")
    for iteration in range(200):
        np.random.seed(seed + iteration)
        np.random.shuffle(sorted_scaffolds)
        train_iter: List[int] = []
        test_iter: List[int] = []
        test_size_current = 0
        for scaffold in sorted_scaffolds:
            indices = scaffold_dict[scaffold]
            if test_size_current + len(indices) <= n_test_target * 1.05:
                test_iter.extend(indices)
                test_size_current += len(indices)
            else:
                train_iter.extend(indices)
        test_labels = y_arr[np.asarray(test_iter, dtype=int)] if test_iter else np.array([], dtype=int)
        pos_test = int(np.sum(test_labels == 1))
        neg_test = int(len(test_labels) - pos_test)
        size_deviation = abs(len(test_iter) - n_test_target) / max(n_test_target, 1)
        pos_deviation = abs(pos_test - target_pos) / max(target_pos, 1) if target_pos > 0 else 0.0
        neg_deviation = abs(neg_test - target_neg) / max(target_neg, 1) if target_neg > 0 else 0.0
        class_penalty = 0.0
        if pos_test < min_pos_required:
            class_penalty += 5.0
        if neg_test < min_neg_required:
            class_penalty += 5.0
        score = size_deviation + 0.75 * pos_deviation + 0.25 * neg_deviation + class_penalty
        if score < best_score:
            best_score = score
            best_train = train_iter
            best_test = test_iter
    assert best_train is not None and best_test is not None
    return np.array(best_train, dtype=int), np.array(best_test, dtype=int)


def split_dataset(smiles: List[str], y: np.ndarray, config: MLQSARConfig, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if config.split_method == "scaffold":
        return scaffold_split(smiles, y, test_size=config.test_size, seed=seed)
    stratify = y if config.task == "classification" else None
    return train_test_split(
        np.arange(len(y)),
        test_size=config.test_size,
        random_state=seed,
        stratify=stratify,
    )


def normalize_augmentation_config(config: MLQSARConfig) -> None:
    aug = config.augmentation if isinstance(config.augmentation, dict) else {}
    normalized = {
        "use_smiles_augmentation": bool(aug.get("use_smiles_augmentation", False)),
        "n_augments": int(aug.get("n_augments", 0)),
        "method": str(aug.get("method", "random_smiles")).strip() or "random_smiles",
        "apply_to": str(aug.get("apply_to", "train_only")).strip() or "train_only",
        "descriptor_mode": str(aug.get("descriptor_mode", "reuse_original")).strip() or "reuse_original",
    }
    if normalized["n_augments"] < 0:
        normalized["n_augments"] = 0
    if normalized["method"] != "random_smiles":
        raise ValueError(f"Unsupported augmentation method: {normalized['method']}")
    if normalized["apply_to"] != "train_only":
        raise ValueError("Only augmentation.apply_to='train_only' is supported")
    if normalized["descriptor_mode"] not in {"recompute", "reuse_original"}:
        raise ValueError("augmentation.descriptor_mode must be 'recompute' or 'reuse_original'")
    config.augmentation = normalized


def normalize_hyperparameter_tuning_config(config: MLQSARConfig) -> None:
    raw = config.hyperparameter_tuning if isinstance(config.hyperparameter_tuning, dict) else {}
    target_models = raw.get("target_models", ["LR", "RFC", "SVC", "XGBC", "ETC", "MLP"])
    if isinstance(target_models, str):
        target_models = [target_models]
    normalized = {
        "enabled": bool(raw.get("enabled", False)),
        "search_type": str(raw.get("search_type", "random")).strip().lower() or "random",
        "target_models": [str(m).strip().upper() for m in target_models if str(m).strip()],
        "n_iter": int(raw.get("n_iter", 60)),
        "cv_folds": int(raw.get("cv_folds", 3)),
        "scoring": str(raw.get("scoring", "roc_auc")).strip() or "roc_auc",
        "n_jobs": int(raw.get("n_jobs", -1)),
    }
    if normalized["search_type"] not in {"random", "grid"}:
        raise ValueError("hyperparameter_tuning.search_type must be 'random' or 'grid'")
    if normalized["n_iter"] <= 0:
        normalized["n_iter"] = 60
    if normalized["cv_folds"] < 2:
        normalized["cv_folds"] = 2
    config.hyperparameter_tuning = normalized


def normalize_thresholding_config(config: MLQSARConfig) -> None:
    raw = config.thresholding if isinstance(config.thresholding, dict) else {}
    threshold = float(raw.get("classification_threshold", 0.5))
    threshold = min(0.99, max(0.01, threshold))
    candidates_raw = raw.get("candidate_thresholds", [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7])
    if isinstance(candidates_raw, (float, int)):
        candidates_raw = [float(candidates_raw)]
    candidates: List[float] = []
    for val in candidates_raw:
        try:
            cand = min(0.99, max(0.01, float(val)))
            candidates.append(cand)
        except Exception:
            continue
    if not candidates:
        candidates = [0.5]
    tune_metric = str(raw.get("tune_metric", "mcc")).strip().lower() or "mcc"
    if tune_metric not in {"mcc", "f1"}:
        tune_metric = "mcc"
    selection_rule = str(raw.get("selection_rule", "max_f1")).strip().lower() or "max_f1"
    if selection_rule not in {"max_f1", "youden_j"}:
        selection_rule = "max_f1"
    curve_points = int(raw.get("curve_points", 201))
    if curve_points < 21:
        curve_points = 21
    if curve_points > 2001:
        curve_points = 2001
    config.thresholding = {
        "enabled": bool(raw.get("enabled", False)),
        "classification_threshold": threshold,
        "tune_in_cv": bool(raw.get("tune_in_cv", False)),
        "tune_metric": tune_metric,
        "candidate_thresholds": sorted(set(candidates)),
        "selection_rule": selection_rule,
        "curve_points": curve_points,
    }


def normalize_descriptor_missing_config(config: MLQSARConfig) -> None:
    raw = config.descriptor_missing if isinstance(config.descriptor_missing, dict) else {}
    strategy = str(raw.get("strategy", "nan_indicator")).strip().lower() or "nan_indicator"
    if strategy not in {"zero", "nan_indicator"}:
        raise ValueError("descriptor_missing.strategy must be 'zero' or 'nan_indicator'")
    impute_value = raw.get("impute_value", 0.0)
    try:
        impute_value_f = float(impute_value)
    except Exception:
        impute_value_f = 0.0
    suffix = str(raw.get("indicator_suffix", "__isna"))
    if not suffix:
        suffix = "__isna"
    config.descriptor_missing = {
        "strategy": strategy,
        "impute_value": impute_value_f,
        "indicator_suffix": suffix,
    }


def compute_morgan_fingerprints(smiles_list: List[str],
                                radius: int = 2,
                                n_bits: int = 2048,
                                show_progress: bool = True) -> np.ndarray:
    fingerprints = []
    generator = MorganGenerator(radius=radius, nBits=n_bits) if _MORGAN_GENERATOR_AVAILABLE else None
    iterator = tqdm(smiles_list, desc="Morgan fingerprints") if show_progress else smiles_list
    for smiles in iterator:
        if not smiles:
            fingerprints.append(np.zeros(n_bits, dtype=np.float32))
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            fingerprints.append(np.zeros(n_bits, dtype=np.float32))
            continue
        if _MORGAN_GENERATOR_AVAILABLE and generator is not None:
            fp = generator.GetFingerprintAsBitVect(mol)
        else:
            fp = GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        fingerprints.append(np.array(fp, dtype=np.float32))
    return np.stack(fingerprints, axis=0)


def compute_rdkit_descriptors(smiles_list: List[str],
                              descriptor_names: List[str],
                              show_progress: bool = True) -> np.ndarray:
    descriptor_funcs = {name: getattr(Descriptors, name) for name in descriptor_names}
    rows = {name: [] for name in descriptor_names}
    iterator = tqdm(smiles_list, desc="RDKit descriptors") if show_progress else smiles_list
    for smiles in iterator:
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        for name, func in descriptor_funcs.items():
            try:
                value = float(func(mol)) if mol is not None else float("nan")
            except Exception:
                value = float("nan")
            rows[name].append(value)
    df = pd.DataFrame(rows)
    # Keep NaN for downstream missing-value handling (imputation + optional indicators).
    return df.astype(np.float32).to_numpy(dtype=np.float32)


def apply_descriptor_missing_strategy(desc_matrix: np.ndarray,
                                      descriptor_names: List[str],
                                      missing_cfg: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
    strategy = str(missing_cfg.get("strategy", "nan_indicator")).strip().lower()
    impute_value = float(missing_cfg.get("impute_value", 0.0))
    suffix = str(missing_cfg.get("indicator_suffix", "__isna")) or "__isna"
    desc = np.asarray(desc_matrix, dtype=np.float32)
    # normalize non-finite values to NaN
    desc[~np.isfinite(desc)] = np.nan
    if strategy == "zero":
        filled = np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        return filled, list(descriptor_names)
    if strategy != "nan_indicator":
        raise ValueError(f"Unsupported descriptor missing strategy: {strategy}")
    missing_ind = np.isnan(desc).astype(np.float32)
    filled = np.nan_to_num(desc, nan=impute_value, posinf=impute_value, neginf=impute_value).astype(np.float32, copy=False)
    feature_names = list(descriptor_names) + [f"{name}{suffix}" for name in descriptor_names]
    return np.concatenate([filled, missing_ind], axis=1).astype(np.float32, copy=False), feature_names


def randomize_smiles(smiles: str, seed: int) -> Optional[str]:
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # RDKit expects seed in a bounded integer range; map deterministically to avoid overflow.
    safe_seed = int(seed) % 2147483647
    if safe_seed <= 0:
        safe_seed = 1
    rdBase.SeedRandomNumberGenerator(safe_seed)
    try:
        return Chem.MolToSmiles(mol, doRandom=True)
    except Exception:
        return None


def augment_training_data(smiles_train: List[str],
                          y_train: np.ndarray,
                          fp_train: np.ndarray,
                          desc_train: np.ndarray,
                          descriptor_names: List[str],
                          descriptor_missing: Dict[str, Any],
                          descriptor_feature_names: List[str],
                          split_seed: int,
                          phase_seed: int,
                          aug_cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    use_aug = bool(aug_cfg.get("use_smiles_augmentation", False))
    n_augments = int(aug_cfg.get("n_augments", 0))
    if not use_aug or n_augments <= 0:
        size = int(len(y_train))
        stats = {
            "enabled": False,
            "original_size": size,
            "augmented_size": size,
            "added_samples": 0,
            "n_augments": n_augments,
            "method": aug_cfg.get("method", "random_smiles"),
            "descriptor_mode": aug_cfg.get("descriptor_mode", "reuse_original"),
            "split_seed": int(split_seed),
            "phase_seed": int(phase_seed),
        }
        groups = np.array([get_scaffold(s) for s in smiles_train], dtype=object)
        return fp_train, desc_train, y_train, groups, stats

    aug_smiles: List[str] = []
    aug_labels: List[int] = []
    aug_desc_rows: List[np.ndarray] = []
    aug_groups: List[str] = []
    base_groups = [get_scaffold(s) for s in smiles_train]
    descriptor_mode = str(aug_cfg.get("descriptor_mode", "reuse_original"))
    for idx, smi in enumerate(smiles_train):
        for aug_idx in range(n_augments):
            det_seed = int(split_seed * 10_000_000 + phase_seed * 100_000 + idx * 100 + aug_idx + 1)
            randomized = randomize_smiles(smi, det_seed)
            if not randomized:
                continue
            aug_smiles.append(randomized)
            aug_labels.append(int(y_train[idx]))
            aug_groups.append(base_groups[idx])
            if descriptor_mode == "reuse_original":
                aug_desc_rows.append(desc_train[idx])

    if not aug_smiles:
        size = int(len(y_train))
        stats = {
            "enabled": True,
            "original_size": size,
            "augmented_size": size,
            "added_samples": 0,
            "n_augments": n_augments,
            "method": aug_cfg.get("method", "random_smiles"),
            "descriptor_mode": descriptor_mode,
            "split_seed": int(split_seed),
            "phase_seed": int(phase_seed),
        }
        groups = np.array(base_groups, dtype=object)
        return fp_train, desc_train, y_train, groups, stats

    fp_aug = compute_morgan_fingerprints(aug_smiles, show_progress=False)
    if descriptor_mode == "reuse_original" and aug_desc_rows:
        desc_aug = np.stack(aug_desc_rows, axis=0).astype(np.float32)
    else:
        desc_aug_raw = compute_rdkit_descriptors(aug_smiles, descriptor_names, show_progress=False)
        desc_aug, aug_feature_names = apply_descriptor_missing_strategy(desc_aug_raw, descriptor_names, descriptor_missing)
        if aug_feature_names != list(descriptor_feature_names):
            raise RuntimeError(
                "Descriptor feature name mismatch after missing-value processing. "
                "Ensure descriptor_missing config and descriptor_names are consistent."
            )
    y_aug = np.asarray(aug_labels, dtype=y_train.dtype)

    fp_out = np.concatenate([fp_train, fp_aug], axis=0).astype(np.float32)
    desc_out = np.concatenate([desc_train, desc_aug], axis=0).astype(np.float32)
    y_out = np.concatenate([y_train, y_aug], axis=0)
    groups_out = np.array(base_groups + aug_groups, dtype=object)
    stats = {
        "enabled": True,
        "original_size": int(len(y_train)),
        "augmented_size": int(len(y_out)),
        "added_samples": int(len(y_out) - len(y_train)),
        "n_augments": n_augments,
        "method": aug_cfg.get("method", "random_smiles"),
        "descriptor_mode": descriptor_mode,
        "split_seed": int(split_seed),
        "phase_seed": int(phase_seed),
    }
    return fp_out, desc_out, y_out, groups_out, stats


def fit_fp_mask(fp_block: np.ndarray,
                variance_threshold: float = 0.05,
                correlation_threshold: float = 0.9) -> np.ndarray:
    assert fp_block.shape[1] == 2048
    variances = np.nanvar(fp_block, axis=0)
    mask = np.ones_like(variances, dtype=bool)
    mask[variances == 0] = False
    mask &= variances >= variance_threshold
    indices = [idx for idx, keep in enumerate(mask) if keep]
    final_mask = np.zeros_like(mask)
    if not indices:
        fallback = int(np.nanargmax(variances))
        final_mask[fallback] = True
        return final_mask
    if len(indices) > 1500:
        print("Warning: large fingerprint block; skipping correlation filtering")
        final_mask[indices] = True
        return final_mask
    subset = fp_block[:, indices]
    corr = np.corrcoef(subset, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    corr = np.abs(corr)
    keep = set(indices)
    for i, idx_i in enumerate(indices):
        if idx_i not in keep:
            continue
        for j in range(i + 1, len(indices)):
            idx_j = indices[j]
            if idx_j not in keep:
                continue
            if corr[i, j] > correlation_threshold:
                vi = variances[idx_i]
                vj = variances[idx_j]
                if vi > vj:
                    keep.discard(idx_j)
                elif vj > vi:
                    keep.discard(idx_i)
                    break
                else:
                    drop = idx_j if idx_j > idx_i else idx_i
                    keep.discard(drop)
                    if drop == idx_i:
                        break
    if not keep:
        fallback = int(np.nanargmax(variances))
        final_mask[fallback] = True
        return final_mask
    for idx in sorted(keep):
        final_mask[idx] = True
    return final_mask


def apply_mask(fp_matrix: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return fp_matrix[:, mask]


def build_models(seed: int) -> Dict[str, Any]:
    return {
        "LR": LogisticRegression(solver="liblinear", random_state=seed, max_iter=1000),
        "RFC": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed),
        "SVC": SVC(kernel="rbf", probability=True, random_state=seed),
        "XGBC": XGBClassifier(n_estimators=200, n_jobs=-1, random_state=seed),
        "ETC": ExtraTreesClassifier(n_estimators=200, n_jobs=-1, random_state=seed),
        "MLP": MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=seed),
    }


def get_hyperparameter_search_space(model_key: str) -> Dict[str, List[Any]]:
    spaces: Dict[str, Dict[str, List[Any]]] = {
        "LR": {
            "C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear"],
        },
        "RFC": {
            "n_estimators": [200, 400, 700],
            "max_depth": [None, 6, 10, 16],
            "min_samples_split": [2, 4, 8, 12],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", "log2", 0.5],
        },
        "XGBC": {
            "n_estimators": [200, 300, 500],
            "max_depth": [3, 4, 6, 8],
            "learning_rate": [0.01, 0.03, 0.1, 0.2],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "min_child_weight": [1, 3, 5],
        },
        "ETC": {
            "n_estimators": [200, 400, 700],
            "max_depth": [None, 6, 10, 16],
            "min_samples_split": [2, 4, 8, 12],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", "log2", 0.5],
        },
        "SVC": {
            "C": [0.1, 1.0, 10.0, 50.0, 100.0],
            "gamma": ["scale", 0.1, 0.01, 0.001],
            "kernel": ["rbf"],
        },
        "MLP": {
            "hidden_layer_sizes": [(128,), (256, 128), (512, 256), (256, 128, 64)],
            "alpha": [1e-5, 1e-4, 1e-3, 1e-2],
            "learning_rate_init": [1e-4, 5e-4, 1e-3, 5e-3],
            "max_iter": [400, 600, 800],
        },
    }
    return spaces.get(model_key, {})


def get_grid_search_space(model_key: str) -> Dict[str, List[Any]]:
    grids: Dict[str, Dict[str, List[Any]]] = {
        "LR": {
            "C": [0.1, 1.0, 10.0],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear"],
        },
        "RFC": {
            "n_estimators": [200, 500],
            "max_depth": [None, 10],
            "min_samples_leaf": [1, 4],
        },
        "ETC": {
            "n_estimators": [200, 500],
            "max_depth": [None, 10],
            "min_samples_leaf": [1, 4],
        },
        "SVC": {
            "C": [1.0, 10.0, 100.0],
            "gamma": ["scale", 0.01, 0.001],
            "kernel": ["rbf"],
        },
        "XGBC": {
            "n_estimators": [200, 500],
            "max_depth": [3, 6],
            "learning_rate": [0.03, 0.1],
            "subsample": [0.8, 1.0],
        },
        "MLP": {
            "hidden_layer_sizes": [(256, 128), (512, 256)],
            "alpha": [1e-4, 1e-3],
            "learning_rate_init": [1e-3, 5e-4],
        },
    }
    return grids.get(model_key, {})


def get_threshold(config: MLQSARConfig) -> float:
    if bool(config.thresholding.get("enabled", False)):
        return float(config.thresholding.get("classification_threshold", 0.5))
    return 0.5


def get_prediction_outputs(model, X: np.ndarray, threshold: float) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)[:, 1]
        except Exception:
            proba = None
    if proba is not None:
        y_pred = (proba >= threshold).astype(int)
        return y_pred, proba
    y_pred = model.predict(X)
    return y_pred, None


def optimize_threshold(y_true: np.ndarray,
                       y_prob: np.ndarray,
                       candidates: List[float],
                       metric: str) -> Tuple[float, float]:
    best_thr = 0.5
    best_score = -float("inf")
    y_true_int = np.asarray(y_true).astype(int)
    y_prob_float = np.asarray(y_prob).astype(float)
    for thr in candidates:
        y_pred = (y_prob_float >= float(thr)).astype(int)
        if metric == "f1":
            score = float(f1_score(y_true_int, y_pred, zero_division=0))
        else:
            score = float(matthews_corrcoef(y_true_int, y_pred))
        if score > best_score:
            best_score = score
            best_thr = float(thr)
    return best_thr, best_score


def compute_confusion_metrics_at_threshold(y_true: np.ndarray,
                                           y_prob: np.ndarray,
                                           threshold: float) -> Dict[str, float]:
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob).astype(float)
    y_pred = (y_prob_arr >= float(threshold)).astype(int)
    tp = int(np.sum((y_true_arr == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true_arr == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true_arr == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true_arr == 1) & (y_pred == 0)))
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    tpr = recall
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    if precision + recall > 0:
        f1 = float(2.0 * precision * recall / (precision + recall))
    else:
        f1 = 0.0
    try:
        mcc = float(matthews_corrcoef(y_true_arr, y_pred))
    except Exception:
        mcc = float("nan")
    return {
        "precision": precision,
        "recall": recall,
        "tpr": tpr,
        "fpr": fpr,
        "f1": f1,
        "mcc": mcc,
    }


def build_threshold_curve_rows(y_true: np.ndarray,
                               y_prob: np.ndarray,
                               thresholds: np.ndarray,
                               split_seed: int,
                               model_key: str,
                               dataset: str,
                               selected_threshold: Optional[float]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob).astype(float)
    for thr in thresholds:
        stats = compute_confusion_metrics_at_threshold(y_true_arr, y_prob_arr, float(thr))
        rows.append({
            "split_seed": split_seed,
            "model": model_key,
            "dataset": dataset,
            "Threshold": float(thr),
            "Precision": stats["precision"],
            "Recall": stats["recall"],
            "TPR": stats["tpr"],
            "FPR": stats["fpr"],
            "F1": stats["f1"],
            "MCC": stats["mcc"],
            "selected_threshold": float(selected_threshold) if selected_threshold is not None else None,
        })
    return rows


def determine_oof_thresholds(y_true: np.ndarray,
                             y_prob: np.ndarray,
                             config: MLQSARConfig) -> Optional[Dict[str, Any]]:
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob).astype(float)
    valid = ~np.isnan(y_prob_arr)
    y_true_arr = y_true_arr[valid]
    y_prob_arr = y_prob_arr[valid]
    if len(y_true_arr) == 0 or np.unique(y_true_arr).shape[0] < 2:
        return None

    fpr, tpr, roc_thresholds = roc_curve(y_true_arr, y_prob_arr)
    finite_roc = np.isfinite(roc_thresholds)
    if np.any(finite_roc):
        fpr = fpr[finite_roc]
        tpr = tpr[finite_roc]
        roc_thresholds = roc_thresholds[finite_roc]
    roc_thresholds = np.clip(roc_thresholds.astype(float), 0.0, 1.0)
    j_scores = tpr - fpr
    j_idx = int(np.nanargmax(j_scores))
    youden_threshold = float(roc_thresholds[j_idx])
    youden_metrics = compute_confusion_metrics_at_threshold(y_true_arr, y_prob_arr, youden_threshold)

    precision, recall, pr_thresholds = precision_recall_curve(y_true_arr, y_prob_arr)
    if len(pr_thresholds) > 0:
        pr_f1 = (2.0 * precision[:-1] * recall[:-1]) / np.maximum(precision[:-1] + recall[:-1], 1e-12)
        f1_idx = int(np.nanargmax(pr_f1))
        max_f1_threshold = float(np.clip(pr_thresholds[f1_idx], 0.0, 1.0))
        max_f1_score = float(pr_f1[f1_idx])
    else:
        max_f1_threshold = youden_threshold
        max_f1_score = float(youden_metrics["f1"])
    max_f1_metrics = compute_confusion_metrics_at_threshold(y_true_arr, y_prob_arr, max_f1_threshold)
    selection_rule = str(config.thresholding.get("selection_rule", "max_f1")).lower()
    selected_threshold = max_f1_threshold if selection_rule == "max_f1" else youden_threshold
    selected_metrics = compute_confusion_metrics_at_threshold(y_true_arr, y_prob_arr, selected_threshold)

    return {
        "selection_rule": selection_rule,
        "selected_threshold": float(selected_threshold),
        "youden_j_threshold": youden_threshold,
        "youden_j_value": float(j_scores[j_idx]),
        "youden_j_f1": float(youden_metrics["f1"]),
        "youden_j_mcc": float(youden_metrics["mcc"]),
        "max_f1_threshold": max_f1_threshold,
        "max_f1_value": max_f1_score,
        "max_f1_mcc": float(max_f1_metrics["mcc"]),
        "selected_f1": float(selected_metrics["f1"]),
        "selected_mcc": float(selected_metrics["mcc"]),
    }


def fit_model_with_optional_tuning(model_key: str,
                                   model,
                                   X_train: np.ndarray,
                                   y_train: np.ndarray,
                                   config: MLQSARConfig,
                                   split_seed: int,
                                   fold_or_phase_seed: int,
                                   groups_train: Optional[np.ndarray] = None,
                                   logger: Optional[logging.Logger] = None):
    tuning_cfg = config.hyperparameter_tuning
    if not bool(tuning_cfg.get("enabled", False)):
        model.fit(X_train, y_train)
        return model, None
    target_models = set(tuning_cfg.get("target_models", []))
    if model_key not in target_models:
        model.fit(X_train, y_train)
        return model, None
    search_type = str(tuning_cfg.get("search_type", "random")).lower()
    if search_type == "grid":
        space = get_grid_search_space(model_key)
    else:
        space = get_hyperparameter_search_space(model_key)
    if not space:
        model.fit(X_train, y_train)
        return model, None
    labels, counts = np.unique(y_train, return_counts=True)
    if len(labels) < 2 or np.min(counts) < 2:
        if logger is not None:
            logger.warning(f"Skipping tuning for {model_key}: insufficient class counts in training fold.")
        model.fit(X_train, y_train)
        return model, None
    cv_folds = min(int(tuning_cfg.get("cv_folds", 3)), int(np.min(counts)))
    if cv_folds < 2:
        model.fit(X_train, y_train)
        return model, None
    groups_arr = np.asarray(groups_train) if groups_train is not None else None
    use_grouped_cv = groups_arr is not None and len(groups_arr) == len(y_train)
    if use_grouped_cv:
        try:
            unique_groups = np.unique(groups_arr)
            if len(unique_groups) < cv_folds:
                cv_folds = len(unique_groups)
            if cv_folds < 2:
                raise ValueError("not enough unique groups for grouped CV tuning")
            cv = StratifiedGroupKFold(
                n_splits=cv_folds,
                shuffle=True,
                random_state=split_seed + fold_or_phase_seed,
            )
            cv_splits = list(cv.split(X=np.zeros(len(y_train)), y=y_train, groups=groups_arr))
            if any(np.unique(y_train[val_idx]).shape[0] < 2 for _, val_idx in cv_splits):
                raise ValueError("single-class validation fold in StratifiedGroupKFold tuning")
        except Exception:
            cv = GroupKFold(n_splits=cv_folds)
            cv_splits = list(cv.split(X=np.zeros(len(y_train)), y=y_train, groups=groups_arr))
    else:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=split_seed + fold_or_phase_seed)
        cv_splits = list(cv.split(X=np.zeros(len(y_train)), y=y_train))
    if search_type == "grid":
        search = GridSearchCV(
            estimator=model,
            param_grid=space,
            scoring=str(tuning_cfg.get("scoring", "roc_auc")),
            n_jobs=int(tuning_cfg.get("n_jobs", -1)),
            cv=cv_splits,
            refit=True,
        )
    else:
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=space,
            n_iter=int(tuning_cfg.get("n_iter", 60)),
            scoring=str(tuning_cfg.get("scoring", "roc_auc")),
            n_jobs=int(tuning_cfg.get("n_jobs", -1)),
            cv=cv_splits,
            random_state=split_seed + fold_or_phase_seed,
            refit=True,
        )
    try:
        search.fit(X_train, y_train)
    except Exception as exc:
        if logger is not None:
            logger.warning(f"Tuning failed for {model_key} ({exc}); using base estimator.")
        model.fit(X_train, y_train)
        return model, None
    tuning_result = {
        "model": model_key,
        "search_type": search_type,
        "best_score": float(search.best_score_),
        "best_params": search.best_params_,
        "n_iter": int(tuning_cfg.get("n_iter", 60)) if search_type == "random" else None,
        "cv_folds": cv_folds,
        "cv_type": "grouped" if use_grouped_cv else "stratified",
    }
    if logger is not None:
        logger.info(f"    Tuning {model_key}: best_score={search.best_score_:.4f}, best_params={search.best_params_}")
    return search.best_estimator_, tuning_result


def enrichment_factor(y_true: np.ndarray, y_score: np.ndarray, top_fraction: float) -> float:
    y_true_arr = np.asarray(y_true).astype(int)
    y_score_arr = np.asarray(y_score).astype(float)
    n_samples = len(y_true_arr)
    if n_samples == 0:
        return float("nan")
    total_hits = int(np.sum(y_true_arr == 1))
    if total_hits == 0:
        return float("nan")
    top_k = max(1, int(np.ceil(float(top_fraction) * n_samples)))
    top_indices = np.argsort(-y_score_arr)[:top_k]
    top_hits = int(np.sum(y_true_arr[top_indices] == 1))
    top_hit_rate = top_hits / float(top_k)
    base_hit_rate = total_hits / float(n_samples)
    if base_hit_rate == 0.0:
        return float("nan")
    return float(top_hit_rate / base_hit_rate)


def normalized_enrichment_factor(y_true: np.ndarray, y_score: np.ndarray, top_fraction: float) -> float:
    y_true_arr = np.asarray(y_true).astype(int)
    n_samples = len(y_true_arr)
    if n_samples == 0:
        return float("nan")
    total_hits = int(np.sum(y_true_arr == 1))
    if total_hits == 0:
        return float("nan")
    top_k = max(1, int(np.ceil(float(top_fraction) * n_samples)))
    ef_val = enrichment_factor(y_true_arr, y_score, top_fraction)
    max_hits = min(total_hits, top_k)
    base_hit_rate = total_hits / float(n_samples)
    ef_max = (max_hits / float(top_k)) / base_hit_rate if base_hit_rate > 0 else float("nan")
    if not np.isfinite(ef_val) or not np.isfinite(ef_max) or ef_max <= 0:
        return float("nan")
    return float(ef_val / ef_max)


def evaluate_classifier(model, X: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred, proba = get_prediction_outputs(model, X, threshold)
    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, y_pred)),
    }
    if proba is not None and len(np.unique(y)) == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y, proba))
        except Exception:
            metrics["roc_auc"] = float("nan")
        try:
            metrics["pr_auc"] = float(average_precision_score(y, proba))
        except Exception:
            metrics["pr_auc"] = float("nan")
        metrics["ef1"] = float(enrichment_factor(y, proba, 0.01))
        metrics["ef5"] = float(enrichment_factor(y, proba, 0.05))
        metrics["ef10"] = float(enrichment_factor(y, proba, 0.10))
        metrics["nef10"] = float(normalized_enrichment_factor(y, proba, 0.10))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
        metrics["ef1"] = float("nan")
        metrics["ef5"] = float("nan")
        metrics["ef10"] = float("nan")
        metrics["nef10"] = float("nan")
    return metrics


def train_and_evaluate(models: List[str],
                       mask: np.ndarray,
                       scaler: Optional[StandardScaler],
                       fp_train: np.ndarray,
                       desc_train: np.ndarray,
                       y_train: np.ndarray,
                       fp_val: np.ndarray,
                       desc_val: np.ndarray,
                       y_val: np.ndarray,
                       descriptor_dim: int,
                       seed: int,
                       phase_seed: int,
                       config: MLQSARConfig,
                       groups_train: Optional[np.ndarray] = None,
                       logger: Optional[logging.Logger] = None) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], List[Dict[str, Any]]]:
    fp_train_filtered = apply_mask(fp_train, mask)
    fp_val_filtered = apply_mask(fp_val, mask)
    desc_train = desc_train.astype(np.float32)
    desc_val = desc_val.astype(np.float32)
    if scaler is not None:
        desc_scaled_train = scaler.transform(desc_train)
        desc_scaled_val = scaler.transform(desc_val)
        assert desc_scaled_train.shape[1] == descriptor_dim
        assert desc_scaled_val.shape[1] == descriptor_dim
        X_train_scaled = np.concatenate([fp_train_filtered, desc_scaled_train], axis=1).astype(np.float32)
        X_val_scaled = np.concatenate([fp_val_filtered, desc_scaled_val], axis=1).astype(np.float32)
    else:
        X_train_scaled = None
        X_val_scaled = None
    X_train_tree = np.concatenate([fp_train_filtered, desc_train], axis=1).astype(np.float32)
    X_val_tree = np.concatenate([fp_val_filtered, desc_val], axis=1).astype(np.float32)
    models_dict = build_models(seed=seed)
    metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    predictions: List[Dict[str, Any]] = []
    threshold = get_threshold(config)
    for model_key in models:
        model = models_dict[model_key]
        if model_key in NON_TREE_MODELS and scaler is not None:
            input_train = X_train_scaled
            input_val = X_val_scaled
        else:
            input_train = X_train_tree
            input_val = X_val_tree
        if input_train is None or input_val is None:
            continue
        model, tuning_result = fit_model_with_optional_tuning(
            model_key=model_key,
            model=model,
            X_train=input_train,
            y_train=y_train,
            config=config,
            split_seed=seed,
            fold_or_phase_seed=phase_seed,
            groups_train=groups_train,
            logger=logger,
        )
        model_threshold = threshold
        metrics[model_key] = {
            "train": evaluate_classifier(model, input_train, y_train, threshold=model_threshold),
            "val": evaluate_classifier(model, input_val, y_val, threshold=model_threshold),
        }
        metrics[model_key]["train"]["classification_threshold"] = float(model_threshold)
        metrics[model_key]["val"]["classification_threshold"] = float(model_threshold)
        if tuning_result is not None:
            metrics[model_key]["tuning"] = tuning_result
        y_pred, y_prob = get_prediction_outputs(model, input_val, model_threshold)
        for idx in range(len(y_val)):
            predictions.append({
                "row_idx": idx,
                "y_true": float(y_val[idx]),
                "y_pred": float(y_pred[idx]),
                "y_prob": float(y_prob[idx]) if y_prob is not None else None,
                "model": model_key,
            })
    return metrics, predictions


def run_cross_validation(config: MLQSARConfig,
                         fp_dev: np.ndarray,
                         desc_dev: np.ndarray,
                         y_dev: np.ndarray,
                         descriptor_dim: int,
                         descriptor_feature_names: List[str],
                         split_seed: int,
                         ids_dev: List[Any],
                         smiles_dev: List[str],
                         split_seed_dir: Path,
                         logger: logging.Logger) -> Tuple[
                             List[Dict[str, Any]],
                             Dict[str, Dict[str, Dict[str, float]]],
                             Dict[str, Any],
                             Dict[str, Dict[str, Any]],
                             List[Dict[str, Any]],
                         ]:
    """Run cross-validation and save results to structured directory."""
    logger.info(f"\nStarting {config.folds}-fold cross-validation on development set (seed={split_seed})")
    groups = np.array([get_scaffold(s) for s in smiles_dev], dtype=object)
    splitter = StratifiedGroupKFold(n_splits=config.folds, shuffle=True, random_state=split_seed)
    try:
        split_pairs = list(splitter.split(X=np.zeros(len(y_dev)), y=y_dev, groups=groups))
        if any(np.unique(y_dev[val_idx]).shape[0] < 2 for _, val_idx in split_pairs):
            raise ValueError("StratifiedGroupKFold produced at least one single-class validation fold")
        logger.info("Using StratifiedGroupKFold with Bemis-Murcko scaffold groups for inner CV")
    except Exception as exc:
        logger.warning(f"StratifiedGroupKFold unavailable/unstable ({exc}); falling back to GroupKFold")
        splitter = GroupKFold(n_splits=config.folds)
        split_pairs = list(splitter.split(X=np.zeros(len(y_dev)), y=y_dev, groups=groups))
        for fold_i, (_, val_idx) in enumerate(split_pairs, start=1):
            val_pos = int(np.sum(y_dev[val_idx] == 1))
            val_total = int(len(val_idx))
            logger.info(
                f"  GroupKFold class balance fold {fold_i}: positives={val_pos}/{val_total} "
                f"({100.0 * val_pos / max(val_total, 1):.1f}%)"
            )
    needs_scaler = any(model in NON_TREE_MODELS for model in config.selected_models)
    fold_results: List[Dict[str, Any]] = []
    all_val_metrics_by_model: Dict[str, List[Dict[str, float]]] = {model: [] for model in config.selected_models}
    augmentation_fold_stats: List[Dict[str, Any]] = []
    tuning_records: List[Dict[str, Any]] = []
    fp_keep_dims: List[int] = []
    all_val_predictions: List[Dict[str, Any]] = []
    
    pred_dir = split_seed_dir / "predictions"
    ensure_dir(pred_dir)
    for fold_idx, (train_idx, val_idx) in enumerate(split_pairs, start=1):
        logger.info(f"  Fold {fold_idx}/{config.folds} | Processing...")
        fp_train, fp_val = fp_dev[train_idx], fp_dev[val_idx]
        desc_train, desc_val = desc_dev[train_idx], desc_dev[val_idx]
        y_train = y_dev[train_idx]
        train_smiles = [smiles_dev[int(i)] for i in train_idx]
        fp_train_aug, desc_train_aug, y_train_aug, groups_train_aug, aug_stats = augment_training_data(
            smiles_train=train_smiles,
            y_train=y_train,
            fp_train=fp_train,
            desc_train=desc_train,
            descriptor_names=config.descriptor_names,
            descriptor_missing=config.descriptor_missing,
            descriptor_feature_names=descriptor_feature_names,
            split_seed=split_seed,
            phase_seed=fold_idx,
            aug_cfg=config.augmentation,
        )
        aug_stats["fold"] = fold_idx
        augmentation_fold_stats.append(aug_stats)
        if aug_stats.get("enabled"):
            logger.info(
                f"    Augmentation fold {fold_idx}: "
                f"{aug_stats['original_size']} -> {aug_stats['augmented_size']} "
                f"(+{aug_stats['added_samples']})"
            )
        mask = fit_fp_mask(fp_train_aug,
                           variance_threshold=config.variance_threshold,
                           correlation_threshold=config.correlation_threshold)
        fp_raw_dim = int(fp_train_aug.shape[1])
        fp_keep_dim = int(mask.sum())
        fp_keep_dims.append(fp_keep_dim)
        logger.info(
            f"    FP filtering (fold {fold_idx}): {fp_raw_dim} -> {fp_keep_dim} "
            f"(retain {100.0 * fp_keep_dim / max(fp_raw_dim, 1):.1f}%)"
        )
        scaler = None
        if needs_scaler:
            scaler = StandardScaler()
            scaler.fit(desc_train_aug.astype(np.float32))
            assert scaler.mean_.shape[0] == descriptor_dim
        metrics, val_predictions = train_and_evaluate(
            config.selected_models,
            mask,
            scaler,
            fp_train_aug,
            desc_train_aug,
            y_train_aug,
            fp_val,
            desc_val,
            y_dev[val_idx],
            descriptor_dim,
            split_seed,
            fold_idx,
            config,
            groups_train_aug,
            logger,
        )
        fold_results.append({"fold": fold_idx, "metrics": metrics})
        for model_key, model_metrics in metrics.items():
            if "val" in model_metrics:
                all_val_metrics_by_model[model_key].append(model_metrics["val"])
            if "tuning" in model_metrics:
                rec = dict(model_metrics["tuning"])
                rec["stage"] = "cv"
                rec["fold"] = fold_idx
                rec["split_seed"] = split_seed
                tuning_records.append(rec)
        for pred in val_predictions:
            global_idx = int(val_idx[pred["row_idx"]])
            pred["id"] = ids_dev[global_idx]
            pred["smiles"] = smiles_dev[global_idx]
            pred["seed"] = split_seed
            pred["fold"] = fold_idx
            pred.pop("row_idx", None)
        if val_predictions:
            fold_pred_df = pd.DataFrame(val_predictions)[
                ["id", "smiles", "y_true", "y_pred", "y_prob", "model", "seed", "fold"]
            ]
            fold_pred_df.to_csv(pred_dir / f"cv_predictions_fold_{fold_idx}.csv", index=False)
            all_val_predictions.extend(val_predictions)
    
    # Aggregate and save CV results
    aggregated = aggregate_fold_results(fold_results)
    
    # Log CV summary table
    val_metrics_summary = {}
    for model_key, model_agg in aggregated.items():
        val_metrics_summary[model_key] = {}
        for metric in EVAL_METRICS:
            stats = model_agg.get(metric, {})
            mean_val = stats.get("mean")
            std_val = stats.get("std")
            if mean_val is not None:
                val_metrics_summary[model_key][metric] = f"{mean_val:.4f} ± {std_val:.4f}"
    
    table_str = format_metrics_table(val_metrics_summary, title="Cross-Validation Summary (Validation Set)")
    logger.info(table_str)
    
    # Save as CSV
    results_dir = split_seed_dir / "results"
    ensure_dir(results_dir)
    fold_df = fold_results_to_dataframe(fold_results)
    fold_df.to_csv(results_dir / "fold_results.csv", index=False)
    
    # Save CV summary as CSV
    cv_summary_df = cv_summary_to_dataframe(aggregated)
    cv_summary_df.to_csv(results_dir / "cv_summary.csv", index=False)
    
    logger.info(f"✓ Saved CV fold results to {results_dir / 'fold_results.csv'}")
    logger.info(f"✓ Saved CV summary to {results_dir / 'cv_summary.csv'}\n")

    default_threshold = get_threshold(config)
    oof_threshold_summary: Dict[str, Dict[str, Any]] = {}
    oof_df = pd.DataFrame(all_val_predictions) if all_val_predictions else pd.DataFrame()
    for model_key in config.selected_models:
        summary: Dict[str, Any] = {
            "selection_rule": str(config.thresholding.get("selection_rule", "max_f1")).lower(),
            "selected_threshold": float(default_threshold),
            "youden_j_threshold": None,
            "youden_j_value": None,
            "youden_j_f1": None,
            "youden_j_mcc": None,
            "max_f1_threshold": None,
            "max_f1_value": None,
            "max_f1_mcc": None,
            "selected_f1": None,
            "selected_mcc": None,
            "n_oof": 0,
        }
        if not oof_df.empty:
            model_df = oof_df[oof_df["model"] == model_key].copy()
            if not model_df.empty and {"y_true", "y_prob"}.issubset(model_df.columns):
                y_true = model_df["y_true"].astype(int).to_numpy()
                y_prob = model_df["y_prob"].astype(float).to_numpy()
                valid = ~np.isnan(y_prob)
                y_true = y_true[valid]
                y_prob = y_prob[valid]
                summary["n_oof"] = int(len(y_prob))
                threshold_summary = determine_oof_thresholds(y_true, y_prob, config)
                if threshold_summary is not None:
                    summary.update(threshold_summary)
                    logger.info(
                        f"OOF threshold candidates [{model_key}] | "
                        f"Youden-J thr={summary['youden_j_threshold']:.4f} (J={summary['youden_j_value']:.4f}, "
                        f"F1={summary['youden_j_f1']:.4f}) | "
                        f"Max-F1 thr={summary['max_f1_threshold']:.4f} (F1={summary['max_f1_value']:.4f}, "
                        f"MCC={summary['max_f1_mcc']:.4f}) | "
                        f"locked={summary['selected_threshold']:.4f} via {summary['selection_rule']}"
                    )
                else:
                    logger.warning(
                        f"OOF threshold selection skipped for {model_key}: "
                        "insufficient class/probability variation; using default threshold."
                    )
        oof_threshold_summary[model_key] = summary

    aug_summary = {
        "enabled": bool(config.augmentation.get("use_smiles_augmentation", False)),
        "folds": augmentation_fold_stats,
        "tuning_records": tuning_records,
        "fp_keep_dims": fp_keep_dims,
    }
    return fold_results, aggregated, aug_summary, oof_threshold_summary, all_val_predictions


def train_final_models(config: MLQSARConfig,
                       fp_dev: np.ndarray,
                       desc_dev: np.ndarray,
                       y_dev: np.ndarray,
                       smiles_dev: List[str],
                       fp_ext: np.ndarray,
                       desc_ext: np.ndarray,
                       y_ext: np.ndarray,
                       descriptor_feature_names: List[str],
                       descriptor_dim: int,
                       split_seed: int,
                       ids_ext: List[Any],
                       smiles_ext: List[str],
                       locked_thresholds: Optional[Dict[str, float]],
                       logger: logging.Logger) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], np.ndarray, Optional[StandardScaler], Dict[str, Any], List[Dict[str, Any]], Dict[str, pd.DataFrame], Dict[str, Any], Dict[str, Dict[str, Any]]]:
    fp_dev_train, desc_dev_train, y_dev_train, groups_dev_train, aug_stats = augment_training_data(
        smiles_train=smiles_dev,
        y_train=y_dev,
        fp_train=fp_dev,
        desc_train=desc_dev,
        descriptor_names=config.descriptor_names,
        descriptor_missing=config.descriptor_missing,
        descriptor_feature_names=descriptor_feature_names,
        split_seed=split_seed,
        phase_seed=9_999,
        aug_cfg=config.augmentation,
    )
    if aug_stats.get("enabled"):
        logger.info(
            "  Augmentation final-dev: "
            f"{aug_stats['original_size']} -> {aug_stats['augmented_size']} "
            f"(+{aug_stats['added_samples']})"
        )
    mask_final = fit_fp_mask(fp_dev_train,
                             variance_threshold=config.variance_threshold,
                             correlation_threshold=config.correlation_threshold)
    fp_raw_dim = int(fp_dev_train.shape[1])
    fp_keep_dim = int(mask_final.sum())
    logger.info(
        f"  FP filtering (final-dev): {fp_raw_dim} -> {fp_keep_dim} "
        f"(retain {100.0 * fp_keep_dim / max(fp_raw_dim, 1):.1f}%)"
    )
    fp_dev_filtered = apply_mask(fp_dev_train, mask_final)
    fp_ext_filtered = apply_mask(fp_ext, mask_final)
    needs_scaler = any(model in NON_TREE_MODELS for model in config.selected_models)
    desc_scaler = None
    desc_dev_scaled = desc_dev_train
    desc_ext_scaled = desc_ext
    if needs_scaler:
        desc_scaler = StandardScaler()
        desc_scaler.fit(desc_dev_train.astype(np.float32))
        assert desc_scaler.mean_.shape[0] == descriptor_dim
        desc_dev_scaled = desc_scaler.transform(desc_dev_train.astype(np.float32))
        desc_ext_scaled = desc_scaler.transform(desc_ext.astype(np.float32))
    X_dev_tree = np.concatenate([fp_dev_filtered, desc_dev_train], axis=1).astype(np.float32)
    X_dev_scaled = np.concatenate([fp_dev_filtered, desc_dev_scaled], axis=1).astype(np.float32)
    X_ext_tree = np.concatenate([fp_ext_filtered, desc_ext], axis=1).astype(np.float32)
    X_ext_scaled = np.concatenate([fp_ext_filtered, desc_ext_scaled], axis=1).astype(np.float32)
    models_dict = build_models(seed=split_seed)
    metrics = {"dev": {}, "external": {}}
    trained_models: Dict[str, Any] = {}
    external_predictions: List[Dict[str, Any]] = []
    tree_importances: Dict[str, pd.DataFrame] = {}
    final_tuning_records: Dict[str, Dict[str, Any]] = {}
    threshold = get_threshold(config)
    threshold_cfg = config.thresholding
    selection_rule = str(threshold_cfg.get("selection_rule", "max_f1")).lower()
    for model_key in config.selected_models:
        model = models_dict[model_key]
        if model_key in NON_TREE_MODELS and needs_scaler:
            model, tuning_result = fit_model_with_optional_tuning(
                model_key=model_key,
                model=model,
                X_train=X_dev_scaled,
                y_train=y_dev_train,
                config=config,
                split_seed=split_seed,
                fold_or_phase_seed=9_999,
                logger=logger,
                groups_train=groups_dev_train,
            )
            train_input = X_dev_scaled
            ext_input = X_ext_scaled
        else:
            model, tuning_result = fit_model_with_optional_tuning(
                model_key=model_key,
                model=model,
                X_train=X_dev_tree,
                y_train=y_dev_train,
                config=config,
                split_seed=split_seed,
                fold_or_phase_seed=9_999,
                logger=logger,
                groups_train=groups_dev_train,
            )
            train_input = X_dev_tree
            ext_input = X_ext_tree
        model_threshold = threshold
        threshold_source = "default"
        if locked_thresholds is not None and model_key in locked_thresholds:
            try:
                model_threshold = float(locked_thresholds[model_key])
                threshold_source = f"oof_{selection_rule}"
            except Exception:
                model_threshold = threshold
                threshold_source = "default"
        logger.info(
            f"  Threshold lock {model_key}: threshold={model_threshold:.4f} "
            f"(source={threshold_source})"
        )
        metrics["dev"][model_key] = evaluate_classifier(model, train_input, y_dev_train, threshold=model_threshold)
        metrics["external"][model_key] = evaluate_classifier(model, ext_input, y_ext, threshold=model_threshold)
        metrics["dev"][model_key]["classification_threshold"] = float(model_threshold)
        metrics["external"][model_key]["classification_threshold"] = float(model_threshold)
        metrics["dev"][model_key]["roc_auc"] = metrics["dev"][model_key].get("roc_auc", float("nan"))
        metrics["external"][model_key]["roc_auc"] = metrics["external"][model_key].get("roc_auc", float("nan"))
        if tuning_result is not None:
            record = dict(tuning_result)
            record["stage"] = "final_dev"
            record["split_seed"] = split_seed
            record["classification_threshold"] = float(model_threshold)
            final_tuning_records[model_key] = record
        trained_models[model_key] = model
        logger.info(f"  ✓ {model_key:8s} | dev AUC: {metrics['dev'][model_key].get('roc_auc', float('nan')):7.4f} | ext AUC: {metrics['external'][model_key].get('roc_auc', float('nan')):7.4f}")
        features_to_use = ext_input
        preds, proba = get_prediction_outputs(model, features_to_use, model_threshold)
        for idx, mol_id in enumerate(ids_ext):
            external_predictions.append({
                "id": mol_id,
                "smiles": smiles_ext[idx],
                "y_true": float(y_ext[idx]),
                "y_pred": float(preds[idx]),
                "y_prob": float(proba[idx]) if proba is not None else None,
                "model": model_key,
                "seed": split_seed,
            })
        if model_key in TREE_MODELS and hasattr(model, "feature_importances_"):
            tree_importances[model_key] = feature_importance_dataframe(
                model.feature_importances_,
                mask_final,
                descriptor_feature_names,
            )
    logger.info(f"\nExternal Test Set Performance:\n")
    ext_metrics_table = {model: metrics["external"][model] for model in config.selected_models}
    ext_table_str = format_metrics_table(ext_metrics_table)
    logger.info(ext_table_str + "\n")
    return metrics, mask_final, desc_scaler, trained_models, external_predictions, tree_importances, aug_stats, final_tuning_records


def create_feature_config(config: MLQSARConfig,
                          mask: np.ndarray,
                          descriptor_feature_names: List[str],
                          descriptor_dim: int,
                          needs_scaler: bool,
                          augmentation_summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    fp_dim_final = int(mask.sum())
    descriptor_start_idx = fp_dim_final
    return {
        "fp_dim_final": fp_dim_final,
        "descriptor_start_idx": descriptor_start_idx,
        "feature_order": "fp_first",
        "fp_params": {"radius": 2, "nBits": 2048},
        "filtering": {
            "variance_threshold": config.variance_threshold,
            "correlation_threshold": config.correlation_threshold,
            "fp_dim_raw": 2048,
            "fp_dim_final": fp_dim_final,
        },
        "descriptor": {
            "raw_names": config.descriptor_names,
            "feature_names": descriptor_feature_names,
            "dim": descriptor_dim,
            "missing": serialize_json(config.descriptor_missing),
        },
        "scaling": {
            "use_for_non_tree": needs_scaler,
            "apply_to_descriptor_only": True,
            "descriptor_start_idx": descriptor_start_idx,
        },
        "augmentation": augmentation_summary or {
            "enabled": False,
            "config": serialize_json(config.augmentation),
        },
        "modeling": {
            "thresholding": serialize_json(config.thresholding),
            "hyperparameter_tuning": serialize_json(config.hyperparameter_tuning),
        },
        "fp_mask_applied_before_concat": True,
    }


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def fingerprint_descriptor_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_fingerprint_descriptor.csv")


def export_fingerprint_descriptor_csv(df: pd.DataFrame,
                                   input_path: Path,
                                   fp_matrix: np.ndarray,
                                   desc_matrix: np.ndarray,
                                   descriptor_names: List[str]) -> Path:
    out_path = fingerprint_descriptor_output_path(input_path)
    export_df = df.copy()
    fp_cols = [f"morgan_{i}" for i in range(fp_matrix.shape[1])]
    for idx, col in enumerate(fp_cols):
        export_df[col] = fp_matrix[:, idx]
    for idx, name in enumerate(descriptor_names):
        export_df[name] = desc_matrix[:, idx]
    export_df.to_csv(out_path, index=False)
    return out_path


def save_model_artifacts(model,
                         model_dir: Path,
                         model_key: str,
                         split_seed: int,
                         feature_config: Dict[str, Any],
                         scaler: Optional[StandardScaler],
                         model_tuning_result: Optional[Dict[str, Any]] = None) -> None:
    ensure_dir(model_dir)
    joblib.dump(model, model_dir / "model.joblib")
    model_config = {
        "model": model_key,
        "split_seed": split_seed,
        "uses_descriptor_scaler": scaler is not None,
        "feature_order": feature_config.get("feature_order", "fp_first"),
        "fp_dim_final": feature_config.get("fp_dim_final"),
        "descriptor_start_idx": feature_config.get("descriptor_start_idx"),
        "thresholding": feature_config.get("modeling", {}).get("thresholding", {}),
        "hyperparameter_tuning": feature_config.get("modeling", {}).get("hyperparameter_tuning", {}),
        "best_hyperparameters": serialize_json(model_tuning_result or {}),
    }
    with open(model_dir / "model_config.json", "w") as fh:
        json.dump(serialize_json(model_config), fh, indent=2)
    if scaler is not None:
        joblib.dump(scaler, model_dir / "scaler.joblib")


def save_split_data(split_seed_dir: Path, fp_dev: np.ndarray, desc_dev: np.ndarray, 
                    fp_ext: np.ndarray, desc_ext: np.ndarray,
                    y_dev: np.ndarray, y_ext: np.ndarray,
                    ids_dev: List[Any], ids_ext: List[Any],
                    smiles_dev: List[str], smiles_ext: List[str]) -> None:
    """Save development and external split arrays with metadata for downstream analysis."""
    data_dir = split_seed_dir / "data" / "splits"
    ensure_dir(data_dir)
    np.savez_compressed(
        data_dir / "dev_train.npz",
        fp=fp_dev.astype(np.float32),
        desc=desc_dev.astype(np.float32),
        y=y_dev.astype(np.int32),
        id=np.asarray(ids_dev, dtype=object),
        smiles=np.asarray(smiles_dev, dtype=object),
    )
    np.savez_compressed(
        data_dir / "external_test.npz",
        fp=fp_ext.astype(np.float32),
        desc=desc_ext.astype(np.float32),
        y=y_ext.astype(np.int32),
        id=np.asarray(ids_ext, dtype=object),
        smiles=np.asarray(smiles_ext, dtype=object),
    )


def save_feature_processors(split_seed_dir: Path, mask: np.ndarray, descriptor_names: List[str],
                           feature_config: Dict[str, Any], mask_fit_on_dev: bool = True) -> None:
    """Save fingerprint mask, descriptor names, and feature configuration."""
    fp_dir = split_seed_dir / "feature_processors"
    ensure_dir(fp_dir)
    np.save(fp_dir / "fp_mask.npy", mask)
    with open(fp_dir / "descriptor_names.json", "w") as f:
        json.dump(descriptor_names, f, indent=2)
    with open(fp_dir / "feature_config.json", "w") as f:
        json.dump(feature_config, f, indent=2)


def select_background_indices(y: np.ndarray, max_samples: int, random_state: int) -> np.ndarray:
    """Select a class-balanced subset when possible for SHAP background data."""
    total = int(len(y))
    if total == 0:
        return np.array([], dtype=int)
    n_take = min(max(1, int(max_samples)), total)
    if n_take >= total:
        return np.arange(total, dtype=int)

    y_arr = np.asarray(y).astype(int)
    rng = np.random.default_rng(int(random_state))
    classes = np.unique(y_arr)
    if len(classes) < 2:
        return np.sort(rng.choice(total, size=n_take, replace=False).astype(int))

    per_class_target = max(1, n_take // len(classes))
    picked: List[int] = []
    for cls in classes:
        cls_idx = np.where(y_arr == cls)[0]
        if len(cls_idx) == 0:
            continue
        take = min(per_class_target, len(cls_idx))
        sampled = rng.choice(cls_idx, size=take, replace=False)
        picked.extend(int(x) for x in sampled)

    if len(picked) < n_take:
        used = set(picked)
        remain = np.array([i for i in range(total) if i not in used], dtype=int)
        if len(remain) > 0:
            extra_take = min(n_take - len(picked), len(remain))
            extra = rng.choice(remain, size=extra_take, replace=False)
            picked.extend(int(x) for x in extra)

    if len(picked) > n_take:
        picked = list(rng.choice(np.array(picked, dtype=int), size=n_take, replace=False))

    return np.sort(np.array(picked, dtype=int))


def save_shap_ready_artifacts(split_seed_dir: Path,
                              selected_models: List[str],
                              mask: np.ndarray,
                              descriptor_names: List[str],
                              scaler: Optional[StandardScaler],
                              fp_dev: np.ndarray,
                              desc_dev: np.ndarray,
                              y_dev: np.ndarray,
                              ids_dev: List[Any],
                              smiles_dev: List[str],
                              fp_ext: np.ndarray,
                              desc_ext: np.ndarray,
                              y_ext: np.ndarray,
                              ids_ext: List[Any],
                              smiles_ext: List[str],
                              split_seed: int,
                              background_max_samples: int = 256) -> Dict[str, Any]:
    """Export SHAP-ready matrices and metadata per model."""
    shap_dir = split_seed_dir / "data" / "shap"
    ensure_dir(shap_dir)
    feature_dir = split_seed_dir / "feature_processors"
    ensure_dir(feature_dir)

    fp_dev_filtered = apply_mask(fp_dev, mask).astype(np.float32)
    fp_ext_filtered = apply_mask(fp_ext, mask).astype(np.float32)
    desc_dev_raw = desc_dev.astype(np.float32)
    desc_ext_raw = desc_ext.astype(np.float32)
    if scaler is not None:
        desc_dev_scaled = scaler.transform(desc_dev_raw).astype(np.float32)
        desc_ext_scaled = scaler.transform(desc_ext_raw).astype(np.float32)
    else:
        desc_dev_scaled = desc_dev_raw
        desc_ext_scaled = desc_ext_raw

    x_dev_tree = np.concatenate([fp_dev_filtered, desc_dev_raw], axis=1).astype(np.float32)
    x_ext_tree = np.concatenate([fp_ext_filtered, desc_ext_raw], axis=1).astype(np.float32)
    x_dev_scaled = np.concatenate([fp_dev_filtered, desc_dev_scaled], axis=1).astype(np.float32)
    x_ext_scaled = np.concatenate([fp_ext_filtered, desc_ext_scaled], axis=1).astype(np.float32)

    feature_names, feature_types = build_feature_names(mask, descriptor_names)
    feature_schema = {
        "feature_names": feature_names,
        "feature_types": feature_types,
        "n_features": int(len(feature_names)),
    }
    with open(feature_dir / "feature_names_final.json", "w") as f:
        json.dump(feature_schema, f, indent=2)

    ids_dev_arr = np.asarray(ids_dev, dtype=object)
    smiles_dev_arr = np.asarray(smiles_dev, dtype=object)
    ids_ext_arr = np.asarray(ids_ext, dtype=object)
    smiles_ext_arr = np.asarray(smiles_ext, dtype=object)
    y_dev_i32 = np.asarray(y_dev).astype(np.int32)
    y_ext_i32 = np.asarray(y_ext).astype(np.int32)

    bg_idx = select_background_indices(y_dev_i32, background_max_samples, random_state=split_seed)
    manifest_rows: List[Dict[str, Any]] = []
    for model_key in selected_models:
        use_scaled = bool(model_key in NON_TREE_MODELS and scaler is not None)
        input_mode = "scaled" if use_scaled else "tree_raw"
        x_dev_model = x_dev_scaled if use_scaled else x_dev_tree
        x_ext_model = x_ext_scaled if use_scaled else x_ext_tree

        bg_path = shap_dir / f"background_{model_key}.npz"
        np.savez_compressed(
            bg_path,
            X=x_dev_model[bg_idx],
            y=y_dev_i32[bg_idx],
            id=ids_dev_arr[bg_idx],
            smiles=smiles_dev_arr[bg_idx],
            feature_names=np.asarray(feature_names, dtype=object),
            feature_types=np.asarray(feature_types, dtype=object),
            model=np.array([model_key], dtype=object),
            input_mode=np.array([input_mode], dtype=object),
        )

        explain_path = shap_dir / f"explain_external_{model_key}.npz"
        np.savez_compressed(
            explain_path,
            X=x_ext_model,
            y=y_ext_i32,
            id=ids_ext_arr,
            smiles=smiles_ext_arr,
            feature_names=np.asarray(feature_names, dtype=object),
            feature_types=np.asarray(feature_types, dtype=object),
            model=np.array([model_key], dtype=object),
            input_mode=np.array([input_mode], dtype=object),
        )

        manifest_rows.append({
            "model": model_key,
            "input_mode": input_mode,
            "background_path": str(bg_path),
            "explain_external_path": str(explain_path),
            "background_n": int(len(bg_idx)),
            "explain_external_n": int(len(y_ext_i32)),
            "n_features": int(x_ext_model.shape[1]),
        })

    manifest = {
        "split_seed": int(split_seed),
        "background_max_samples": int(background_max_samples),
        "rows": manifest_rows,
    }
    with open(shap_dir / "shap_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def fold_results_to_dataframe(fold_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert fold results to DataFrame."""
    rows = []
    for result in fold_results:
        fold_idx = result["fold"]
        for model_key, model_metrics in result["metrics"].items():
            # Validation metrics
            val_metrics = model_metrics.get("val", {})
            row = {"fold": fold_idx, "model": model_key, "split": "val"}
            for metric in EVAL_METRICS:
                row[metric] = val_metrics.get(metric)
            rows.append(row)
            # Train metrics
            train_metrics = model_metrics.get("train", {})
            row = {"fold": fold_idx, "model": model_key, "split": "train"}
            for metric in EVAL_METRICS:
                row[metric] = train_metrics.get(metric)
            rows.append(row)
    return pd.DataFrame(rows)


def external_predictions_to_dataframe(external_preds: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert external predictions to DataFrame."""
    df = pd.DataFrame(external_preds)
    return df


def cv_summary_to_dataframe(cv_summary: Dict[str, Dict[str, Dict[str, float]]]) -> pd.DataFrame:
    """Convert aggregated CV summary to DataFrame."""
    rows = []
    for model_key, model_summary in cv_summary.items():
        for metric, stats in model_summary.items():
            rows.append({
                "model": model_key,
                "metric": metric,
                "mean": stats.get("mean"),
                "std": stats.get("std"),
                "mean_std": (
                    f"{stats.get('mean'):.4f} ± {stats.get('std'):.4f}"
                    if stats.get("mean") is not None and stats.get("std") is not None
                    else None
                ),
            })
    return pd.DataFrame(rows)


def ensure_run_dir(config: MLQSARConfig) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = config.output_root / f"qsar_ml_{timestamp}"
    ensure_dir(run_dir)
    return run_dir


def write_publication_methods_summary(run_dir: Path,
                                      config: MLQSARConfig,
                                      config_source: str,
                                      n_samples: int,
                                      n_active: int,
                                      n_inactive: int,
                                      precomputed_fp: bool,
                                      precomputed_desc: bool) -> Path:
    report_path = run_dir / "results" / "publication_methods_summary.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    threshold_cfg = config.thresholding if isinstance(config.thresholding, dict) else {}
    tuning_cfg = config.hyperparameter_tuning if isinstance(config.hyperparameter_tuning, dict) else {}
    aug_cfg = config.augmentation if isinstance(config.augmentation, dict) else {}
    descriptor_preview = ", ".join(config.descriptor_names[:6])
    if len(config.descriptor_names) > 6:
        descriptor_preview += ", ..."
    text = (
        "1) Data and Feature Configuration\n"
        f"- Input file: {Path(config.input_path).resolve()}\n"
        f"- Sample counts: total={n_samples}, active={n_active}, inactive={n_inactive}\n"
        f"- Split strategy: method={config.split_method}, external_test_size={config.test_size}, "
        f"split_seeds={config.split_seeds}, cv_folds={config.folds}\n"
        f"- Morgan fingerprints: 2048 bits, source={'reused input columns' if precomputed_fp else 'recomputed by RDKit'}\n"
        f"- RDKit descriptors: n={len(config.descriptor_names)}, source={'reused input columns' if precomputed_desc else 'recomputed by RDKit'}\n"
        f"- Descriptor preview: {descriptor_preview}\n\n"
        "2) Models and Training Strategy\n"
        f"- Models: {', '.join(config.selected_models)}\n"
        f"- FP filtering: variance_threshold={config.variance_threshold}, correlation_threshold={config.correlation_threshold}\n"
        "- Feature concatenation: fp_first (filtered fingerprint block + descriptor block)\n"
        "- Scaling policy: descriptor-only scaling for LR/SVC/MLP; raw descriptors for tree models\n"
        f"- Hyperparameter tuning: enabled={bool(tuning_cfg.get('enabled', False))}, "
        f"search_type={tuning_cfg.get('search_type', 'random')}, "
        f"cv_folds={tuning_cfg.get('cv_folds', 3)}, scoring={tuning_cfg.get('scoring', 'roc_auc')}\n"
        f"- Thresholding: enabled={bool(threshold_cfg.get('enabled', False))}, "
        f"selection_rule={threshold_cfg.get('selection_rule', 'max_f1')}\n"
        "- Threshold selection protocol: compute Youden-J and Max-F1 on OOF predictions, "
        "lock one threshold, then apply it directly to external test (no threshold search on test set)\n"
        f"- Augmentation: enabled={bool(aug_cfg.get('use_smiles_augmentation', False))}, "
        f"n_augments={aug_cfg.get('n_augments', 0)}, descriptor_mode={aug_cfg.get('descriptor_mode', 'reuse_original')}\n\n"
        "3) Outputs and Reproducibility\n"
        f"- Run directory: {run_dir.resolve()}\n"
        "- Per-seed outputs: CV metrics (fold_results.csv, cv_summary.csv), "
        "external metrics/predictions (external_test_results.csv, external_test_predictions.csv)\n"
        "- Threshold outputs: threshold_selection_summary.csv, threshold_curves_data.csv\n"
        "- Artifacts: model.joblib, scaler.joblib, fp_mask.npy, feature_config.json\n"
        "- Aggregated outputs: results/all_seed_*\n"
        f"- Effective config source: {config_source}\n"
        "- Full effective parameters (JSON):\n"
        f"{json.dumps(serialize_json(asdict(config)), ensure_ascii=False, indent=2)}\n"
    )
    report_path.write_text(text, encoding="utf-8")
    return report_path


def configure_logger(run_dir: Path) -> logging.Logger:
    log_dir = run_dir / "logs"
    ensure_dir(log_dir)
    log_path = log_dir / "run.log"
    logger = logging.getLogger("step011")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    stream = logging.StreamHandler()
    stream.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(stream)
    logger.info("Logger initialized")
    return logger


def run_seed(split_seed: int,
             config: MLQSARConfig,
             run_dir: Path,
             fp_matrix: np.ndarray,
             desc_matrix: np.ndarray,
             descriptor_feature_names: List[str],
             y: np.ndarray,
             smiles: List[str],
             ids: List[Any],
             logger: logging.Logger) -> Dict[str, Any]:
    """Run training for a single split seed and organize outputs hierarchically."""
    print_process_divider(logger, f"Processing Split Seed {split_seed}")
    split_seed_dir = run_dir / f"split_seed_{split_seed}"
    ensure_dir(split_seed_dir)
    
    # Split data
    dev_idx, ext_idx = split_dataset(smiles, y, config, split_seed)
    fp_dev, fp_ext = fp_matrix[dev_idx], fp_matrix[ext_idx]
    desc_dev, desc_ext = desc_matrix[dev_idx], desc_matrix[ext_idx]
    y_dev, y_ext = y[dev_idx], y[ext_idx]
    ids_dev = [ids[i] for i in dev_idx]
    ids_ext = [ids[i] for i in ext_idx]
    smiles_dev = [smiles[i] for i in dev_idx]
    smiles_ext = [smiles[i] for i in ext_idx]
    descriptor_dim = desc_matrix.shape[1]
    logger.info(f"Data split: {len(dev_idx)} development | {len(ext_idx)} external test")
    dev_pos = int(np.sum(y_dev == 1))
    ext_pos = int(np.sum(y_ext == 1))
    dev_total = int(len(y_dev))
    ext_total = int(len(y_ext))
    logger.info(
        "Class balance | dev: "
        f"{dev_pos}/{dev_total} active ({100.0 * dev_pos / max(dev_total, 1):.1f}%) | "
        f"external: {ext_pos}/{ext_total} active ({100.0 * ext_pos / max(ext_total, 1):.1f}%)"
    )
    if ext_pos < 1 or (ext_total - ext_pos) < 1:
        logger.warning(
            "External split has a single class; scaffold split should be revisited "
            "or test_size/seeds adjusted."
        )
    
    # Run cross-validation
    cv_fold_results, cv_summary, cv_aug_summary, oof_threshold_summary, oof_predictions = run_cross_validation(
        config,
        fp_dev,
        desc_dev,
        y_dev,
        descriptor_dim,
        descriptor_feature_names,
        split_seed,
        ids_dev,
        smiles_dev,
        split_seed_dir,
        logger,
    )
    locked_thresholds = {
        model_key: float(summary.get("selected_threshold", get_threshold(config)))
        for model_key, summary in oof_threshold_summary.items()
    }
    
    # Train final models
    final_metrics, mask_final, scaler, trained_models, external_predictions, tree_importances, final_aug_stats, final_tuning_records = train_final_models(
        config,
        fp_dev,
        desc_dev,
        y_dev,
        smiles_dev,
        fp_ext,
        desc_ext,
        y_ext,
        descriptor_feature_names,
        descriptor_dim,
        split_seed,
        ids_ext,
        smiles_ext,
        locked_thresholds,
        logger,
    )
    final_fp_kept = int(mask_final.sum())
    
    # Create feature configuration
    needs_scaler = any(model in NON_TREE_MODELS for model in config.selected_models)
    feature_config = create_feature_config(
        config,
        mask_final,
        descriptor_feature_names,
        descriptor_dim,
        needs_scaler,
        augmentation_summary={
            "enabled": bool(config.augmentation.get("use_smiles_augmentation", False)),
            "config": serialize_json(config.augmentation),
            "cv": serialize_json(cv_aug_summary),
            "final_dev": serialize_json(final_aug_stats),
        },
    )
    
    # Save feature processors
    save_feature_processors(split_seed_dir, mask_final, descriptor_feature_names, feature_config)
    
    # Save split data
    save_split_data(
        split_seed_dir,
        fp_dev,
        desc_dev,
        fp_ext,
        desc_ext,
        y_dev,
        y_ext,
        ids_dev,
        ids_ext,
        smiles_dev,
        smiles_ext,
    )
    
    # Save trained models
    models_dir = split_seed_dir / "models" / "full_dev"
    ensure_dir(models_dir)
    for model_key, model in trained_models.items():
        model_path = models_dir / model_key / f"seed_{split_seed}"
        ensure_dir(model_path)
        save_model_artifacts(
            model,
            model_path,
            model_key,
            split_seed,
            feature_config,
            scaler if model_key in NON_TREE_MODELS and needs_scaler else None,
            final_tuning_records.get(model_key),
        )
    
    # Save external test predictions
    external_test_dir = split_seed_dir / "predictions"
    ensure_dir(external_test_dir)
    if external_predictions:
        ext_preds_df = external_predictions_to_dataframe(external_predictions)
        ext_preds_df = ext_preds_df[["id", "smiles", "y_true", "y_pred", "y_prob", "model", "seed"]]
        ext_preds_df.to_csv(external_test_dir / "external_test_predictions.csv", index=False)
        logger.info(f"  ✓ Saved external predictions")
    else:
        ext_preds_df = pd.DataFrame(columns=["id", "smiles", "y_true", "y_pred", "y_prob", "model", "seed"])

    # Save external test metrics
    results_dir = split_seed_dir / "results"
    ensure_dir(results_dir)
    cv_tuning_records = cv_aug_summary.get("tuning_records", []) if isinstance(cv_aug_summary, dict) else []
    tuning_summary_payload = {
        "split_seed": split_seed,
        "cv_tuning_records": cv_tuning_records,
        "final_dev_tuning_records": final_tuning_records,
    }
    with open(results_dir / "hyperparameter_tuning_summary.json", "w") as fh:
        json.dump(serialize_json(tuning_summary_payload), fh, indent=2)
    threshold_rows: List[Dict[str, Any]] = []
    for model_key, summary in oof_threshold_summary.items():
        row = {"model": model_key}
        row.update(serialize_json(summary))
        threshold_rows.append(row)
    if threshold_rows:
        pd.DataFrame(threshold_rows).to_csv(results_dir / "threshold_selection_summary.csv", index=False)
        logger.info("  ✓ Saved threshold_selection_summary.csv")
    external_result_rows = []
    for model_key, metrics_dict in final_metrics.get("external", {}).items():
        row = {"model": model_key}
        for metric in EVAL_METRICS:
            row[metric] = metrics_dict.get(metric)
        external_result_rows.append(row)
    if external_result_rows:
        external_results_df = pd.DataFrame(external_result_rows)
        external_results_df.to_csv(results_dir / "external_test_results.csv", index=False)
        summary_rows = []
        for _, row in external_results_df.iterrows():
            summary = {"model": row["model"]}
            for metric in EVAL_METRICS:
                value = row.get(metric)
                summary[f"{metric}_mean"] = value
                summary[f"{metric}_std"] = 0.0 if value is not None and not pd.isna(value) else None
                summary[f"{metric}_mean_std"] = (
                    f"{float(value):.4f} ± 0.0000"
                    if value is not None and not pd.isna(value)
                    else None
                )
            summary_rows.append(summary)
        pd.DataFrame(summary_rows).to_csv(results_dir / "external_test_summary.csv", index=False)
        logger.info("  ✓ Saved external_test_results.csv and external_test_summary.csv")

    threshold_points = int(config.thresholding.get("curve_points", 201))
    thresholds = np.linspace(0.0, 1.0, threshold_points)
    curve_rows: List[Dict[str, Any]] = []
    oof_df = pd.DataFrame(oof_predictions) if oof_predictions else pd.DataFrame()
    for model_key in config.selected_models:
        selected_threshold = locked_thresholds.get(model_key, get_threshold(config))
        if not oof_df.empty:
            oof_model_df = oof_df[oof_df["model"] == model_key]
            if not oof_model_df.empty:
                y_true_oof = oof_model_df["y_true"].astype(int).to_numpy()
                y_prob_oof = oof_model_df["y_prob"].astype(float).to_numpy()
                valid_oof = ~np.isnan(y_prob_oof)
                y_true_oof = y_true_oof[valid_oof]
                y_prob_oof = y_prob_oof[valid_oof]
                if len(y_true_oof) > 0:
                    curve_rows.extend(
                        build_threshold_curve_rows(
                            y_true=y_true_oof,
                            y_prob=y_prob_oof,
                            thresholds=thresholds,
                            split_seed=split_seed,
                            model_key=model_key,
                            dataset="oof",
                            selected_threshold=selected_threshold,
                        )
                    )
        if not ext_preds_df.empty:
            ext_model_df = ext_preds_df[ext_preds_df["model"] == model_key]
            if not ext_model_df.empty:
                y_true_ext = ext_model_df["y_true"].astype(int).to_numpy()
                y_prob_ext = ext_model_df["y_prob"].astype(float).to_numpy()
                valid_ext = ~np.isnan(y_prob_ext)
                y_true_ext = y_true_ext[valid_ext]
                y_prob_ext = y_prob_ext[valid_ext]
                if len(y_true_ext) > 0:
                    curve_rows.extend(
                        build_threshold_curve_rows(
                            y_true=y_true_ext,
                            y_prob=y_prob_ext,
                            thresholds=thresholds,
                            split_seed=split_seed,
                            model_key=model_key,
                            dataset="external",
                            selected_threshold=selected_threshold,
                        )
                    )
    if curve_rows:
        pd.DataFrame(curve_rows).to_csv(results_dir / "threshold_curves_data.csv", index=False)
        logger.info("  ✓ Saved threshold_curves_data.csv")
    
    # Save feature importance for tree models
    if tree_importances:
        interp_dir = split_seed_dir / "interpretation"
        ensure_dir(interp_dir)
        for model_key, feat_df in tree_importances.items():
            model_interp_dir = interp_dir / model_key
            ensure_dir(model_interp_dir)
            feat_df.to_csv(model_interp_dir / "feature_importance.csv", index=False)
        logger.info(f"  ✓ Saved feature importance for {len(tree_importances)} tree models")
    
    # Save split indices
    with open(split_seed_dir / "split_indices.json", "w") as f:
        json.dump(serialize_json({
            "train": dev_idx.tolist(),
            "external": ext_idx.tolist(),
        }), f, indent=2)

    shap_manifest = save_shap_ready_artifacts(
        split_seed_dir=split_seed_dir,
        selected_models=config.selected_models,
        mask=mask_final,
        descriptor_names=descriptor_feature_names,
        scaler=scaler,
        fp_dev=fp_dev,
        desc_dev=desc_dev,
        y_dev=y_dev,
        ids_dev=ids_dev,
        smiles_dev=smiles_dev,
        fp_ext=fp_ext,
        desc_ext=desc_ext,
        y_ext=y_ext,
        ids_ext=ids_ext,
        smiles_ext=smiles_ext,
        split_seed=split_seed,
    )
    logger.info(
        f"  ✓ Saved SHAP-ready artifacts: {len(shap_manifest.get('rows', []))} model bundles "
        f"under {split_seed_dir / 'data' / 'shap'}"
    )
    
    logger.info(f"  ✓ Saved split indices")
    logger.info(f"\n✓ Completed split_seed {split_seed} | Dev: {len(dev_idx)} | External: {len(ext_idx)}\n")
    
    return {
        "split_seed": split_seed,
        "cv_fold_results": cv_fold_results,
        "cv_summary": cv_summary,
        "cv_augmentation": cv_aug_summary,
        "cv_tuning_records": cv_tuning_records,
        "final_tuning_records": final_tuning_records,
        "final_dev_augmentation": final_aug_stats,
        "final_metrics": final_metrics,
        "external_predictions": external_predictions,
        "tree_importances": tree_importances,
        "feature_filtering": {
            "cv_fp_keep_dims": cv_aug_summary.get("fp_keep_dims", []) if isinstance(cv_aug_summary, dict) else [],
            "final_fp_kept": final_fp_kept,
            "fp_raw_dim": int(fp_matrix.shape[1]),
        },
        "oof_threshold_summary": oof_threshold_summary,
        "split_indices": {
            "train": dev_idx.tolist(),
            "external": ext_idx.tolist(),
        },
        "shap_manifest": shap_manifest,
    }


def main() -> None:
    args = parse_args()
    base_config = MLQSARConfig()
    config_source = "default configuration"
    if args.config is not None:
        config_source = f"config file: {Path(args.config).resolve()}"
    file_overrides = load_config_file(args.config)
    for key, value in file_overrides.items():
        if hasattr(base_config, key):
            setattr(base_config, key, value)
    if args.input:
        base_config.input_path = args.input
    if args.output_root:
        base_config.output_root = args.output_root
    if args.models:
        base_config.selected_models = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.seeds:
        base_config.split_seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    if args.test_size is not None:
        base_config.test_size = args.test_size
    if args.split_method:
        base_config.split_method = args.split_method
    if args.folds:
        base_config.folds = args.folds
    if args.variance_threshold is not None:
        base_config.variance_threshold = args.variance_threshold
    if args.correlation_threshold is not None:
        base_config.correlation_threshold = args.correlation_threshold
    if args.descriptor_names:
        base_config.descriptor_names = [name.strip() for name in args.descriptor_names.split(",") if name.strip()]
    cli_overrides = []
    if args.input:
        cli_overrides.append("input")
    if args.output_root:
        cli_overrides.append("output_root")
    if args.models:
        cli_overrides.append("models")
    if args.seeds:
        cli_overrides.append("seeds")
    if args.test_size is not None:
        cli_overrides.append("test_size")
    if args.split_method:
        cli_overrides.append("split_method")
    if args.folds:
        cli_overrides.append("folds")
    if args.variance_threshold is not None:
        cli_overrides.append("variance_threshold")
    if args.correlation_threshold is not None:
        cli_overrides.append("correlation_threshold")
    if args.descriptor_names:
        cli_overrides.append("descriptor_names")
    if cli_overrides:
        if args.config is not None:
            config_source += f" + CLI overrides: {', '.join(cli_overrides)}"
        else:
            config_source = f"default + CLI overrides: {', '.join(cli_overrides)}"
    if not base_config.split_seeds:
        base_config.split_seeds = [base_config.seed + i for i in range(3)]
    normalize_augmentation_config(base_config)
    normalize_hyperparameter_tuning_config(base_config)
    normalize_thresholding_config(base_config)
    normalize_descriptor_missing_config(base_config)
    base_config.input_path = Path(base_config.input_path)
    base_config.output_root = Path(base_config.output_root)
    base_config.output_root.mkdir(parents=True, exist_ok=True)
    run_dir = ensure_run_dir(base_config)
    logger = configure_logger(run_dir)
    log_runtime_info(logger)
    
    print_process_divider(logger, "QSAR ML Pipeline Initialization")
    logger.info(f"Seeds: {base_config.split_seeds}")
    logger.info(f"Models: {', '.join(base_config.selected_models)}")
    logger.info(f"Split method: {base_config.split_method} | Test size: {base_config.test_size}")
    logger.info(f"Augmentation: {json.dumps(serialize_json(base_config.augmentation), ensure_ascii=True)}")
    logger.info(f"Hyperparameter tuning: {json.dumps(serialize_json(base_config.hyperparameter_tuning), ensure_ascii=True)}")
    logger.info(f"Thresholding: {json.dumps(serialize_json(base_config.thresholding), ensure_ascii=True)}")
    logger.info(f"Output: {run_dir}\n")
    
    print_process_divider(logger, "Data Loading & Pre-processing")
    df = read_table(base_config.input_path)
    logger.info(f"Loaded data: {len(df)} molecules")
    if base_config.smiles_column not in df.columns:
        raise ValueError(f"Missing SMILES column: {base_config.smiles_column}")
    if base_config.label_column not in df.columns:
        raise ValueError(f"Missing label column: {base_config.label_column}")
    if base_config.id_column not in df.columns:
        logger.warning(f"ID column '{base_config.id_column}' not found; using row indices")
        df[base_config.id_column] = df.index.astype(str)
    smiles = df[base_config.smiles_column].fillna("").astype(str).tolist()
    ids = df[base_config.id_column].fillna("").astype(str).tolist()
    labels = pd.Series(df[base_config.label_column]).astype(str).str.strip().str.lower()
    mapper = {"1": 1, "0": 0, "true": 1, "false": 0, "active": 1, "inactive": 0}
    if set(labels.unique()) - set(mapper.keys()):
        numeric = pd.to_numeric(df[base_config.label_column], errors="coerce")
        if numeric.isnull().any():
            raise ValueError("NaN detected in label column after numeric conversion")
        y = numeric.to_numpy(dtype=int)
    else:
        y = labels.map(mapper).to_numpy(dtype=int)
    logger.info(f"Parsed labels: {np.sum(y)} active | {len(y) - np.sum(y)} inactive")
    
    print_process_divider(logger, "Feature Engineering")
    fp_matrix = detect_existing_fingerprints(df)
    precomputed_fp = fp_matrix is not None
    if fp_matrix is not None:
        logger.info("✓ Using pre-existing Morgan fingerprint columns from input")
    else:
        logger.info("Computing Morgan fingerprints via RDKit...")
        fp_matrix = compute_morgan_fingerprints(smiles)
    descriptors_precomputed = set(base_config.descriptor_names).issubset(df.columns)
    if descriptors_precomputed:
        logger.info("✓ Using provided descriptor columns from input")
        desc_matrix_raw = df[base_config.descriptor_names].astype(np.float32).to_numpy(dtype=np.float32)
    else:
        logger.info("Computing RDKit descriptors...")
        desc_matrix_raw = compute_rdkit_descriptors(smiles, base_config.descriptor_names)

    desc_matrix, descriptor_feature_names = apply_descriptor_missing_strategy(
        desc_matrix_raw,
        base_config.descriptor_names,
        base_config.descriptor_missing,
    )

    all_features_precomputed = precomputed_fp and descriptors_precomputed
    if all_features_precomputed:
        current_path = str(base_config.input_path.resolve())
        logger.info(f"Feature/Fingerprint reuse enabled. Current file: {current_path}")
        print(f"Feature/Fingerprint file (reused): {current_path}")
    else:
        out_csv = export_fingerprint_descriptor_csv(
            df=df,
            input_path=base_config.input_path,
            fp_matrix=fp_matrix,
            desc_matrix=desc_matrix,
            descriptor_names=descriptor_feature_names,
        )
        out_path_str = str(out_csv.resolve())
        logger.info(f"Recomputed features exported to: {out_path_str}")
        print(f"Feature/Fingerprint file (recomputed): {out_path_str}")
    
    print_feature_audit(
        logger,
        fp_raw=int(fp_matrix.shape[1]),
        fp_kept=None,
        descriptors_used=descriptor_feature_names,
        precomputed_found=precomputed_fp,
    )
    
    print_process_divider(logger, "Model Training")
    seed_results: List[Dict[str, Any]] = []
    for split_seed in base_config.split_seeds:
        seed_results.append(
            run_seed(
                split_seed,
                base_config,
                run_dir,
                fp_matrix,
                desc_matrix,
                descriptor_feature_names,
                y,
                smiles,
                ids,
                logger,
            )
        )

    print_process_divider(logger, "Aggregating Results Across Seeds")
    # Aggregate results across all seeds
    all_cv_fold_results = []
    all_external_preds: List[Dict[str, Any]] = []
    all_tuning_rows: List[Dict[str, Any]] = []
    external_metrics_by_model: Dict[str, Dict[str, List[float]]] = {}
    feature_filtering_rows: List[Dict[str, Any]] = []
    
    for result in seed_results:
        all_cv_fold_results.extend(result.get("cv_fold_results", []))
        all_external_preds.extend(result.get("external_predictions", []))
        for rec in result.get("cv_tuning_records", []):
            row = {
                "split_seed": result["split_seed"],
                "stage": rec.get("stage", "cv"),
                "fold": rec.get("fold"),
                "model": rec.get("model"),
                "best_score": rec.get("best_score"),
                "cv_type": rec.get("cv_type"),
                "cv_folds": rec.get("cv_folds"),
                "n_iter": rec.get("n_iter"),
                "classification_threshold": rec.get("classification_threshold"),
                "best_params_json": json.dumps(serialize_json(rec.get("best_params", {})), ensure_ascii=True),
            }
            all_tuning_rows.append(row)
        for model_key, rec in result.get("final_tuning_records", {}).items():
            row = {
                "split_seed": result["split_seed"],
                "stage": rec.get("stage", "final_dev"),
                "fold": None,
                "model": rec.get("model", model_key),
                "best_score": rec.get("best_score"),
                "cv_type": rec.get("cv_type"),
                "cv_folds": rec.get("cv_folds"),
                "n_iter": rec.get("n_iter"),
                "classification_threshold": rec.get("classification_threshold"),
                "best_params_json": json.dumps(serialize_json(rec.get("best_params", {})), ensure_ascii=True),
            }
            all_tuning_rows.append(row)
        for model_key, metrics in result["final_metrics"]["external"].items():
            external_metrics_by_model.setdefault(model_key, {metric: [] for metric in EVAL_METRICS})
            for metric in EVAL_METRICS:
                val = metrics.get(metric)
                if val is not None:
                    external_metrics_by_model[model_key][metric].append(float(val))
        ff = result.get("feature_filtering", {})
        cv_dims = ff.get("cv_fp_keep_dims", [])
        fp_raw_dim = ff.get("fp_raw_dim", 2048)
        for fold_i, keep_dim in enumerate(cv_dims, start=1):
            feature_filtering_rows.append({
                "split_seed": result["split_seed"],
                "stage": "cv",
                "fold": fold_i,
                "fp_raw_dim": fp_raw_dim,
                "fp_kept_dim": int(keep_dim),
                "retention_rate": float(keep_dim) / max(float(fp_raw_dim), 1.0),
            })
        if ff.get("final_fp_kept") is not None:
            final_keep = int(ff["final_fp_kept"])
            feature_filtering_rows.append({
                "split_seed": result["split_seed"],
                "stage": "final_dev",
                "fold": None,
                "fp_raw_dim": fp_raw_dim,
                "fp_kept_dim": final_keep,
                "retention_rate": float(final_keep) / max(float(fp_raw_dim), 1.0),
            })
    # Create global results directory
    global_results_dir = run_dir / "results"
    ensure_dir(global_results_dir)
    
    # Aggregate CV results across all seeds
    if all_cv_fold_results:
        cv_aggregated = aggregate_fold_results(all_cv_fold_results)
        cv_summary_df = cv_summary_to_dataframe(cv_aggregated)
        cv_summary_df.to_csv(global_results_dir / "all_seed_cv_summary.csv", index=False)
        logger.info(f"✓ Saved aggregated CV summary")
    
    # Aggregate external test results across all seeds
    if all_external_preds:
        all_external_df = pd.DataFrame(all_external_preds)
        all_external_df.to_csv(global_results_dir / "all_seed_external_predictions.csv", index=False)
        logger.info(f"✓ Saved all external predictions ({len(all_external_preds)} total predictions)")
    
    # Aggregate hyperparameter tuning records across all seeds
    if all_tuning_rows:
        all_tuning_df = pd.DataFrame(all_tuning_rows)
        all_tuning_df.to_csv(global_results_dir / "all_seed_hyperparameter_tuning_summary.csv", index=False)
        logger.info(f"✓ Saved hyperparameter tuning summary ({len(all_tuning_rows)} records)")

    if feature_filtering_rows:
        feature_filtering_df = pd.DataFrame(feature_filtering_rows)
        feature_filtering_df.to_csv(global_results_dir / "feature_filtering_summary.csv", index=False)
        final_rows = feature_filtering_df[feature_filtering_df["stage"] == "final_dev"]
        if not final_rows.empty:
            logger.info(
                "Feature filtering (final-dev) kept bits: "
                f"{final_rows['fp_kept_dim'].mean():.1f} ± {final_rows['fp_kept_dim'].std(ddof=0):.1f}"
            )
        logger.info("✓ Saved feature filtering summary (per-fold and final-dev)")
    
    # Create summary metrics across seeds (mean ± std)
    summary_metrics: Dict[str, Any] = {}
    for model_key, metric_dict in external_metrics_by_model.items():
        summary_metrics[model_key] = {"metrics": {}}
        for metric, values in metric_dict.items():
            if not values:
                summary_metrics[model_key]["metrics"][metric] = {"mean": None, "std": None, "n": 0}
                continue
            summary_metrics[model_key]["metrics"][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "n": len(values),
            }
    
    # Save summary metrics
    with open(global_results_dir / "summary_metrics.json", "w") as fh:
        json.dump(serialize_json(summary_metrics), fh, indent=2)
    logger.info(f"✓ Saved summary metrics (mean ± std across {len(base_config.split_seeds)} seeds)")
    
    # Create external test summary CSV (mean ± std across seeds)
    external_summary_rows = []
    for model_key, model_summary in summary_metrics.items():
        row = {"model": model_key}
        for metric, stats in model_summary["metrics"].items():
            if stats["mean"] is not None:
                row[f"{metric}_mean"] = stats["mean"]
                row[f"{metric}_std"] = stats["std"]
                row[f"{metric}_mean_std"] = f"{stats['mean']:.4f} ± {stats['std']:.4f}"
            else:
                row[f"{metric}_mean"] = None
                row[f"{metric}_std"] = None
                row[f"{metric}_mean_std"] = None
        external_summary_rows.append(row)
    
    if external_summary_rows:
        external_summary_df = pd.DataFrame(external_summary_rows)
        external_summary_df.to_csv(global_results_dir / "all_seed_external_summary.csv", index=False)
        logger.info(f"✓ Saved external test summary")

    try:
        publication_summary_path = write_publication_methods_summary(
            run_dir=run_dir,
            config=base_config,
            config_source=config_source,
            n_samples=int(len(df)),
            n_active=int(np.sum(y)),
            n_inactive=int(len(y) - np.sum(y)),
            precomputed_fp=bool(precomputed_fp),
            precomputed_desc=bool(descriptors_precomputed),
        )
        logger.info(f"✓ Saved publication methods summary: {publication_summary_path}")
    except Exception as exc:
        logger.warning(f"Failed to save publication methods summary: {exc}")
    
    print_process_divider(logger, f"Run Complete - {len(base_config.split_seeds)} Seeds")
    logger.info(f"Models trained: {', '.join(base_config.selected_models)}")
    logger.info(f"All results organized under: {run_dir}\n")
    print(f"\n✓ Run complete. Output directory: {run_dir}\n")


if __name__ == "__main__":
    main()
