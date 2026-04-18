#!/usr/bin/env python3
"""
Production-grade QSAR virtual screening inference with Applicability Domain (AD) scoring.

This script performs large-scale inference over a precomputed feature table
(`zinc_features.parquet`) using artifacts produced by `scripts/step10_qsar_ml.py`
and integrates real-time Applicability Domain (AD) scoring from `scripts/step22_applicability_domain.py`.

Critical alignment rules implemented here
- Load and validate the training feature schema from:
  - `feature_processors/feature_names_final.json`
  - `feature_processors/descriptor_names.json`
  - `feature_processors/fp_mask.npy` (optional, but validated if present)
- Enforce exact feature ordering as in training (fp-first, then descriptors).
- Descriptors:
  - Use raw descriptor values (float32)
  - Do NOT create `__isna` indicator features
  - Do NOT impute; missing values remain NaN
  - Rows with NaN descriptors are skipped (model inputs cannot contain NaN)
- Fingerprints:
  - Use precomputed Morgan bits from input parquet (`morgan_0..morgan_2047`)
  - Dtype stays uint8 for storage; cast to float32 only for model input

Applicability Domain (AD) Integration
- Load AD artifacts: StandardScaler, PCA (95% variance), training set features
- Real-time AD scoring within batch loop:
  - Leverage calculation using pre-trained PCA
  - Maximum Tanimoto similarity to training set
  - Maximum Cosine similarity to training set
  - Weighted AD score using ad_weight_config.json
  - Power law transformation: AD_Score = (fused)^power
- Optimized similarity search using sklearn.neighbors.NearestNeighbors

I/O + performance
- Streaming read with `pyarrow.parquet.ParquetFile.iter_batches`
- Streaming write with `pyarrow.parquet.ParquetWriter` (zstd)
- tqdm progress bar over batches
- Memory-stable processing using Polars and batching

python scripts/step33_vs_inference.py \
  --model_dir ./models_out/qsar_ml_20260412_162829/split_seed_12345 \
  --model_name SVC \
  --seed 12345 \
  --calibration isotonic \
  --threshold auto \
  --threshold_metric mcc \
  --input ./data/database/zinc_features.parquet \
  --ad_integration

  # threshold optional: f1(default currently), youden, mcc, recall, precision, or specific value\

[Original liagnd (A1A0M) validation]
python scripts/step33_vs_inference.py \
  --model_dir ./models_out/qsar_ml_20260412_162829/split_seed_12345 \
  --model_name SVC \
  --seed 12345 \
  --calibration isotonic \
  --threshold auto \
  --threshold_metric mcc \
  --input ./docking/9CVD/a1a0m_final.parquet \
  --output ./docking/9CVD/a1a0m_final_inference_result.parquet \
  --ad_integration

python scripts/step33_vs_inference.py \
  --model_dir ./models_out/qsar_ml_20260412_162829/split_seed_12345 \
  --model_name SVC \
  --seed 12345 \
  --calibration sigmoid \
  --threshold auto \
  --threshold_metric mcc \
  --input ./models_out/qsar_ml_20260412_162829/original_ligand_QSAR/A1A0M_feature.parquet \
  --output ./models_out/qsar_ml_20260412_162829/original_ligand_QSAR/A1A0M_inference_result_sigmoid.parquet \
  --ad_integration

[Inference for NSD2 development set]
python scripts/step33_vs_inference.py \
  --model_dir ./models_out/qsar_ml_20260412_162829/split_seed_12345 \
  --model_name SVC \
  --seed 12345 \
  --calibration isotonic \
  --threshold auto \
  --threshold_metric mcc \
  --input ./data/NSD2/nsd2_dev_set_seed12345.parquet \
  --output ./data/NSD2/dev_set_validation_result.parquet \
  --ad_integration

[Inference for NSD2 External test set]
python scripts/step33_vs_inference.py \
  --model_dir ./models_out/qsar_ml_20260412_162829/split_seed_12345 \
  --model_name SVC \
  --seed 12345 \
  --calibration isotonic \
  --threshold auto \
  --threshold_metric mcc \
  --input ./data/NSD2/nsd2_test_set_seed12345.parquet \
  --output ./data/NSD2/test_set_inference_result.parquet \
  --ad_integration

"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import argparse
import json
import os
import logging  # Keep logging available throughout the module
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
import joblib
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

def _require_deps() -> None:
    try:
        import pyarrow
        import tqdm
        import sklearn
    except Exception as exc:
        raise SystemExit(f"Missing runtime dependencies: {exc}")

try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None  # type: ignore[assignment]


NON_TREE_MODELS: Set[str] = {"LR", "SVC", "MLP"}


@dataclass(frozen=True)
class ArtifactPaths:
    run_dir: Path
    split_dir: Path
    feature_names_path: Path
    descriptor_names_path: Path
    fp_mask_path: Path
    model_path: Path
    scaler_path: Path
    calibrated_model_path: Optional[Path]
    threshold_summary_path: Path


@dataclass(frozen=True)
class ADArtifacts:
    """Container for Applicability Domain artifacts."""
    pca: Any  # PCA model for leverage calculation
    pca_scaler: Any  # Scaler used for PCA projection
    train_features: Any  # Training set features for similarity search
    train_fingerprints: Any  # Training set fingerprints for Tanimoto similarity
    ad_weight_config: Dict[str, float]  # AD weight configuration
    leverage_pca_variance: float  # PCA variance ratio used
    ad_score_power: float  # Power exponent for AD score


@dataclass(frozen=True)
class FeaturePlan:
    feature_names_final: List[str]  # exact order used in training
    descriptor_names: List[str]
    fp_indices: List[int]  # kept fingerprint indices in order
    fp_input_columns: List[str]  # mapped parquet columns: morgan_<idx>

    @property
    def n_fp(self) -> int:
        return len(self.fp_indices)

    @property
    def n_desc(self) -> int:
        return len(self.descriptor_names)

    @property
    def n_features_total(self) -> int:
        return self.n_fp + self.n_desc


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Industrial-scale QSAR virtual screening inference (Parquet streaming)")
    p.add_argument(
        "--model_dir",
        type=Path,
        required=True,
        help="Run directory from step10 (contains split_seed_*) OR a split_seed_* directory.",
    )
    p.add_argument("--model_name", type=str, required=True, choices=["ETC", "RFC", "XGBC", "SVC", "LR", "MLP"])
    p.add_argument("--seed", type=int, required=True, help="Split seed used during training (e.g. 12345)")
    p.add_argument("--calibration", type=str, default="none", choices=["isotonic", "sigmoid", "none"])
    p.add_argument("--threshold", type=str, default="auto", help='Float value or "auto"')
    p.add_argument("--batch_size", type=int, default=100_000)
    p.add_argument(
        "--input",
        type=Path,
        default=Path("data/database/zinc_features.parquet"),
        help="Feature table parquet (must contain zinc_id, smiles, and feature columns).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output parquet path. Default: <run_dir>/virtual_screening/zinc_predictions_<timestamp>.parquet",
    )
    p.add_argument(
        "--threshold_metric",
        type=str,
        default="f1",
        choices=["f1", "youden", "mcc"],
        help="When --threshold auto, which metric to use for threshold selection (f1/youden/mcc).",
    )
    p.add_argument(
        "--smiles_validation",
        type=str,
        default="rdkit",
        choices=["rdkit", "none"],
        help="SMILES validation strategy. 'rdkit' skips invalid SMILES (CPU cost).",
    )
    p.add_argument(
        "--ad_integration",
        action="store_true",
        default=False,
        help="Enable Applicability Domain (AD) integration. Requires step22 AD artifacts.",
    )
    return p.parse_args(argv)


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _safe_float(s: Union[str, float, int]) -> float:
    try:
        return float(s)
    except Exception as exc:
        raise ValueError(f"Invalid float value: {s!r}") from exc


def resolve_run_and_split_dirs(model_dir: Path, seed: int) -> Tuple[Path, Path]:
    model_dir = model_dir.resolve()
    if model_dir.name.startswith("split_seed_"):
        split_dir = model_dir
        run_dir = model_dir.parent
        if split_dir.name != f"split_seed_{seed}":
            raise ValueError(
                f"--seed ({seed}) does not match provided split directory name ({split_dir.name})."
            )
        return run_dir, split_dir

    split_dir = model_dir / f"split_seed_{seed}"
    if split_dir.exists():
        return model_dir, split_dir

    split_dirs = sorted([p for p in model_dir.glob("split_seed_*") if p.is_dir()])
    hint = ", ".join(p.name for p in split_dirs[:10]) + (" ..." if len(split_dirs) > 10 else "")
    raise FileNotFoundError(
        f"Could not find split directory {split_dir}. Available under {model_dir}: {hint or '(none)'}"
    )


def build_artifact_paths(model_dir: Path, model_name: str, seed: int, calibration: str) -> ArtifactPaths:
    run_dir, split_dir = resolve_run_and_split_dirs(model_dir, seed)

    feature_names_path = split_dir / "feature_processors" / "feature_names_final.json"
    descriptor_names_path = split_dir / "feature_processors" / "descriptor_names.json"
    fp_mask_path = split_dir / "feature_processors" / "fp_mask.npy"

    model_path = split_dir / "models" / "full_dev" / model_name / f"seed_{seed}" / "model.joblib"
    scaler_path = split_dir / "models" / "full_dev" / model_name / f"seed_{seed}" / "scaler.joblib"
    threshold_summary_path = split_dir / "results" / "threshold_selection_summary.csv"

    calibrated_model_path: Optional[Path] = None
    if calibration != "none":
        calibrated_model_path = (
            split_dir / "calibration" / model_name / f"method_{calibration}" / "calibrated_model.joblib"
        )

    return ArtifactPaths(
        run_dir=run_dir,
        split_dir=split_dir,
        feature_names_path=feature_names_path,
        descriptor_names_path=descriptor_names_path,
        fp_mask_path=fp_mask_path,
        model_path=model_path,
        scaler_path=scaler_path,
        calibrated_model_path=calibrated_model_path,
        threshold_summary_path=threshold_summary_path,
    )


def _resolve_ad_seed_dir(ad_root: Path, model_name: str, seed: int) -> Path:
    candidates = [
        ad_root / "validation" / "applicability_domain" / model_name / f"seed_{seed}",
        ad_root / model_name / f"seed_{seed}",
        ad_root / f"seed_{seed}",
        ad_root,
    ]
    for candidate in candidates:
        if candidate.exists() and (candidate / "ad_pca_model.joblib").exists():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve AD artifact directory from {ad_root}. "
        f"Expected a folder containing ad_pca_model.joblib for model={model_name}, seed={seed}."
    )


def _find_split_dir_from_ad_dir(ad_dir: Path) -> Path:
    for candidate in [ad_dir, *ad_dir.parents]:
        if candidate.name.startswith("split_seed_"):
            return candidate
    raise FileNotFoundError(f"Could not locate split_seed_* ancestor for AD directory: {ad_dir}")


def load_ad_artifacts(ad_root: Path, model_name: str, seed: int, plan: FeaturePlan):
    import logging

    logger = logging.getLogger(__name__)
    artifacts: Dict[str, Any] = {}

    try:
        ad_dir = _resolve_ad_seed_dir(Path(ad_root), model_name=model_name, seed=seed)
        split_dir = _find_split_dir_from_ad_dir(ad_dir)

        config_file = ad_dir / "ad_weight_config.json"
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                artifacts["weights"] = json.load(f)
            logger.info("Loaded AD weight config")
        else:
            logger.warning(f"ad_weight_config.json missing at {ad_dir}")

        summary_file = ad_dir / "ad_summary.json"
        if summary_file.exists():
            summary = _read_json(summary_file)
            if isinstance(summary, dict):
                artifacts["base_method"] = str(summary.get("base_method", "leverage")).strip().lower()
                continuous = summary.get("continuous_ad_scores", {})
                if isinstance(continuous, dict):
                    artifacts["ad_score_power"] = float(continuous.get("ad_score_power", 2.0))

        pca_file = ad_dir / "ad_pca_model.joblib"
        scaler_file = ad_dir / "ad_pca_scaler.joblib"
        if pca_file.exists():
            artifacts["pca"] = joblib.load(pca_file)
        if scaler_file.exists():
            artifacts["scaler"] = joblib.load(scaler_file)

        npz_file = ad_dir / "ad_plot_data.npz"
        if npz_file.exists():
            data = np.load(npz_file, allow_pickle=True)
            if "desc_train_scaled" in data:
                artifacts["train_desc_scaled"] = np.asarray(data["desc_train_scaled"], dtype=np.float32)
            logger.info("Loaded AD plot reference arrays")

        train_npz = split_dir / "data" / "splits" / "dev_train.npz"
        if not train_npz.exists():
            raise FileNotFoundError(f"Missing dev_train.npz required for AD inference: {train_npz}")

        train = np.load(train_npz, allow_pickle=True)
        train_fp = np.asarray(train["fp"], dtype=np.float32)
        train_desc = np.asarray(train["desc"], dtype=np.float32)

        fp_mask_path = split_dir / "feature_processors" / "fp_mask.npy"
        if fp_mask_path.exists():
            fp_mask = np.load(fp_mask_path).astype(bool)
            if fp_mask.ndim != 1 or fp_mask.shape[0] != train_fp.shape[1]:
                raise ValueError(
                    f"Unexpected fp_mask shape {fp_mask.shape}; expected ({train_fp.shape[1]},)"
                )
            train_fp = train_fp[:, fp_mask]
        else:
            if train_fp.shape[1] >= max(plan.fp_indices) + 1:
                train_fp = train_fp[:, plan.fp_indices]
            elif train_fp.shape[1] != plan.n_fp:
                raise ValueError(
                    f"Cannot align train fingerprints to plan. train_fp shape={train_fp.shape}, expected n_fp={plan.n_fp}"
                )

        if train_fp.shape[1] != plan.n_fp:
            raise ValueError(f"AD train fingerprint dim mismatch: {train_fp.shape[1]} vs expected {plan.n_fp}")
        if train_desc.shape[1] != plan.n_desc:
            raise ValueError(f"AD train descriptor dim mismatch: {train_desc.shape[1]} vs expected {plan.n_desc}")

        train_base_raw = np.concatenate([train_fp, train_desc], axis=1).astype(np.float32, copy=False)
        if train_base_raw.shape[1] != plan.n_features_total:
            raise ValueError(
                f"AD train feature dim mismatch: {train_base_raw.shape[1]} vs expected {plan.n_features_total}"
            )

        artifacts["n_fp"] = int(plan.n_fp)
        artifacts["n_desc"] = int(plan.n_desc)
        artifacts["train_fp_bin"] = np.clip(np.round(train_fp).astype(np.int8), 0, 1)

        if "train_desc_scaled" not in artifacts:
            from sklearn.preprocessing import StandardScaler

            desc_scaler = StandardScaler()
            artifacts["train_desc_scaled"] = desc_scaler.fit_transform(train_desc).astype(np.float32)

        pca = artifacts.get("pca")
        ad_scaler = artifacts.get("scaler")
        if pca is not None and ad_scaler is not None:
            train_base_scaled = ad_scaler.transform(train_base_raw).astype(np.float32, copy=False)
            train_base_pca = pca.transform(train_base_scaled)
            xtx_inv = np.linalg.pinv(train_base_pca.T @ train_base_pca)
            train_leverage = np.einsum("ij,jk,ik->i", train_base_pca, xtx_inv, train_base_pca)
            finite_train_leverage = train_leverage[np.isfinite(train_leverage) & (train_leverage >= 0.0)]
            if finite_train_leverage.size == 0:
                raise ValueError("Failed to compute finite training leverage reference values.")
            artifacts["train_pca_xtx_inv"] = xtx_inv
            artifacts["density_reference_median"] = float(np.median(finite_train_leverage))

        logger.info(
            "AD artifacts ready: n_fp=%s n_desc=%s density_ref=%.6f",
            artifacts.get("n_fp"),
            artifacts.get("n_desc"),
            float(artifacts.get("density_reference_median", np.nan)),
        )

    except Exception as exc:
        logger.error(f"Critical error loading AD artifacts: {exc}")
        return None

    return artifacts


def load_model(paths: ArtifactPaths, model_name: str, calibration: str):
    if joblib is None:  # pragma: no cover
        raise SystemExit("joblib is required to load model artifacts (model.joblib / scaler.joblib).")

    model = None
    used_calibration = "none"
    if calibration != "none" and paths.calibrated_model_path is not None and paths.calibrated_model_path.exists():
        model = joblib.load(paths.calibrated_model_path)
        used_calibration = calibration
    else:
        if calibration != "none":
            print(f"[Warn] Calibrated model not found; falling back to raw model. Expected: {paths.calibrated_model_path}")
        if not paths.model_path.exists():
            raise FileNotFoundError(f"Base model not found: {paths.model_path}")
        model = joblib.load(paths.model_path)

    scaler = None
    if model_name in NON_TREE_MODELS:
        if not paths.scaler_path.exists():
            raise FileNotFoundError(f"Scaler expected for {model_name} but missing: {paths.scaler_path}")
        scaler = joblib.load(paths.scaler_path)
    else:
        scaler = None

    if not hasattr(model, "predict_proba"):
        raise TypeError(f"Loaded model does not support predict_proba: {type(model)}")

    return model, scaler, used_calibration


def load_feature_plan(paths: ArtifactPaths) -> FeaturePlan:
    schema = _read_json(paths.feature_names_path)
    if not isinstance(schema, dict) or "feature_names" not in schema:
        raise ValueError(f"Unexpected feature_names_final.json format: {paths.feature_names_path}")
    feature_names = schema["feature_names"]
    if not isinstance(feature_names, list) or not all(isinstance(x, str) for x in feature_names):
        raise ValueError(f"Invalid 'feature_names' in: {paths.feature_names_path}")

    if not paths.descriptor_names_path.exists():
        raise FileNotFoundError(f"Missing descriptor_names.json: {paths.descriptor_names_path}")
    descriptor_names = _read_json(paths.descriptor_names_path)
    if not isinstance(descriptor_names, list) or not all(isinstance(x, str) for x in descriptor_names) or not descriptor_names:
        raise ValueError(f"Invalid descriptor_names.json content: {paths.descriptor_names_path}")

    # Enforce "no __isna" policy for this production script per requirements.
    forbidden = [n for n in feature_names if n.endswith("__isna")]
    if forbidden:
        raise ValueError(
            "Training feature schema contains '__isna' features, but this inference script is configured to forbid them. "
            f"Found: {forbidden[:10]}{' ...' if len(forbidden) > 10 else ''}"
        )

    fp_indices: List[int] = []
    descriptor_part: List[str] = []
    for name in feature_names:
        if name.startswith("fp_") and name.split("fp_", 1)[-1].isdigit():
            fp_indices.append(int(name.split("fp_", 1)[-1]))
        else:
            descriptor_part.append(name)

    if not fp_indices:
        raise ValueError("No fingerprint features detected in feature_names_final.json (expected fp_<idx>).")

    if descriptor_part != list(descriptor_names):
        raise ValueError(
            "Descriptor feature ordering mismatch between feature_names_final.json and descriptor_names.json.\n"
            f"  feature_names_final descriptor tail (n={len(descriptor_part)}): {descriptor_part[:8]}...\n"
            f"  descriptor_names.json (n={len(descriptor_names)}): {list(descriptor_names)[:8]}..."
        )

    fp_input_columns = [f"morgan_{i}" for i in fp_indices]
    return FeaturePlan(
        feature_names_final=list(feature_names),
        descriptor_names=list(descriptor_names),
        fp_indices=fp_indices,
        fp_input_columns=fp_input_columns,
    )


def validate_fp_mask(paths: ArtifactPaths, plan: FeaturePlan) -> None:
    if not paths.fp_mask_path.exists():
        print(f"[Info] fp_mask.npy not found at {paths.fp_mask_path}; continuing without mask validation.")
        return
    import numpy as np

    mask = np.load(paths.fp_mask_path)
    if mask.ndim != 1 or mask.shape[0] != 2048:
        raise ValueError(f"Unexpected fp_mask shape {mask.shape} in {paths.fp_mask_path}")
    kept = [int(i) for i in np.where(mask)[0].tolist()]
    if kept != plan.fp_indices:
        raise ValueError(
            "Fingerprint mask mismatch: fp_mask.npy indices do not match fp_* features in feature_names_final.json.\n"
            f"  fp_mask kept (n={len(kept)}): {kept[:10]}{' ...' if len(kept) > 10 else ''}\n"
            f"  feature_names fp (n={len(plan.fp_indices)}): {plan.fp_indices[:10]}{' ...' if len(plan.fp_indices) > 10 else ''}"
        )


def _normalize_smiles_series(smiles_col) -> "pandas.Series":
    s = smiles_col
    if not isinstance(s, pd.Series):
        s = pd.Series(s)
    # Keep as python strings; empty/None are invalid.
    s = s.astype("string")
    return s


def _compute_leverage_pca(
    pca: Any,
    pca_scaler: Any,
    features: np.ndarray,
    train_pca_xtx_inv: Optional[np.ndarray],
) -> np.ndarray:
    """Compute PCA leverage with the same hat-matrix formula used in step22."""
    if train_pca_xtx_inv is None:
        raise ValueError("train_pca_xtx_inv is required for exact PCA leverage computation.")

    features_scaled = pca_scaler.transform(features)
    features_pca = pca.transform(features_scaled)
    leverage = np.einsum("ij,jk,ik->i", features_pca, train_pca_xtx_inv, features_pca)
    return leverage.astype(np.float32)


def _compute_tanimoto_similarity(train_fp: np.ndarray, query_fp: np.ndarray) -> np.ndarray:
    """Compute maximum Tanimoto similarity between query and training fingerprints."""
    # Binarize fingerprints
    train_bin = np.clip(np.round(train_fp).astype(np.int8), 0, 1)
    query_bin = np.clip(np.round(query_fp).astype(np.int8), 0, 1)
    
    # Compute max Tanimoto similarity for each query
    max_similarities = np.zeros(len(query_bin), dtype=np.float32)
    
    for i, q_fp in enumerate(query_bin):
        # Compute intersection and union
        intersection = np.sum(train_bin & q_fp, axis=1)
        union = np.sum(train_bin | q_fp, axis=1)
        
        # Avoid division by zero
        union = np.where(union == 0, 1, union)
        similarities = intersection / union
        
        max_similarities[i] = np.max(similarities) if similarities.size > 0 else 0.0
    
    return max_similarities


def _compute_cosine_similarity(train_features: np.ndarray, query_features: np.ndarray, 
                              block_size: int = 1024) -> np.ndarray:
    """Compute maximum cosine similarity between query and training features (blockwise)."""
    eps = 1e-12
    
    # Normalize training features
    train_norm = np.linalg.norm(train_features, axis=1, keepdims=True)
    train_norm = np.where(train_norm < eps, 1.0, train_norm)
    train_unit = train_features / train_norm
    
    max_similarities = np.zeros(len(query_features), dtype=np.float32)
    
    # Process in blocks to reduce memory usage
    for start in range(0, len(query_features), block_size):
        end = min(len(query_features), start + block_size)
        q_block = query_features[start:end]
        
        # Normalize query block
        q_norm = np.linalg.norm(q_block, axis=1, keepdims=True)
        q_norm = np.where(q_norm < eps, 1.0, q_norm)
        q_unit = q_block / q_norm
        
        # Compute cosine similarities
        similarities = q_unit @ train_unit.T
        max_similarities[start:end] = np.max(similarities, axis=1)
    
    return max_similarities


def _compute_density_score_from_reference(
    density_arr: np.ndarray,
    reference_median: Optional[float],
    eps: float = 1e-12,
) -> np.ndarray:
    """Normalize density-like values against a stable training reference."""
    density_arr = np.asarray(density_arr, dtype=np.float64)
    density_arr = np.where(
        np.isnan(density_arr) | np.isinf(density_arr) | (density_arr < 0.0),
        0.0,
        density_arr,
    )

    d0 = float(reference_median) if reference_median is not None else float(np.median(density_arr))
    d0 = max(d0, eps)
    density_score = 1.0 / (1.0 + (density_arr / d0))
    return np.clip(density_score, 0.0, 1.0).astype(np.float32)


def _compute_ad_score(
    density_score: np.ndarray,
    max_tanimoto: np.ndarray,
    max_cosine: np.ndarray,
    ad_config: Dict[str, float],
) -> np.ndarray:
    """Compute final AD score using weighted similarity + density fusion."""
    w1 = ad_config.get("w1_tanimoto", 0.7)
    w2 = ad_config.get("w2_cosine", 0.3)
    w3 = ad_config.get("w3_similarity", 0.6)
    w4 = ad_config.get("w4_density", 0.4)
    power = ad_config.get("ad_score_power", 2.0)

    similarity_score = w1 * max_tanimoto + w2 * max_cosine
    similarity_score = np.clip(similarity_score, 0.0, 1.0)

    ad_raw = w3 * similarity_score + w4 * density_score
    ad_raw = np.clip(ad_raw, 0.0, 1.0)
    ad_final = np.power(np.clip(ad_raw, 1e-7, 1.0), power)
    return ad_final.astype(np.float32)


def select_required_input_columns(parquet_schema_names: List[str], plan: FeaturePlan) -> List[str]:
    import logging
    logger = logging.getLogger(__name__)
    
    available_columns = parquet_schema_names
    names_set = set(available_columns)
    
    logger.info("="*50)
    logger.info("VALIDATING INPUT DATA SCHEMA")
    logger.info(f"Input parquet contains {len(available_columns)} columns.")
    logger.info(f"Sample columns from input: {available_columns[:10]}...")

    # 1. Auto-detect the identifier column.
    id_col = None
    possible_id_names = ["zinc_id", "id", "ZINC_ID", "compound_id"]
    for name in possible_id_names:
        if name in names_set:
            id_col = name
            break
    
    if id_col:
        logger.info(f"âœ… Found ID column: '{id_col}' (will be mapped to 'zinc_id')")
    else:
        logger.error(f"âŒ CRITICAL: No ID column found. Looked for: {possible_id_names}")
        raise KeyError(f"Input parquet must contain an ID column. Available: {available_columns[:20]}")

    # 2. Check that the SMILES column is available.
    if "smiles" not in names_set:
        logger.error("âŒ CRITICAL: 'smiles' column missing!")
        raise KeyError("Input parquet must contain a 'smiles' column for AD/Inference.")

    required_cols_set = {id_col, "smiles"}

    # 3. Validate fingerprint columns.
    missing_fps = []
    for col in plan.fp_input_columns:
        if col not in names_set:
            missing_fps.append(col)
        else:
            required_cols_set.add(col)
    
    if missing_fps:
        logger.error(f"âŒ CRITICAL: {len(missing_fps)} fingerprint columns missing!")
        logger.error(f"First few missing: {missing_fps[:5]}")
        raise KeyError(f"Missing required fingerprint column: {missing_fps[0]}")
    else:
        logger.info(f"âœ… All {len(plan.fp_input_columns)} fingerprint columns present.")

    # 4. Validate descriptor columns.
    missing_descs = []
    for feat in plan.descriptor_names:
        if feat not in names_set:
            missing_descs.append(feat)
        else:
            required_cols_set.add(feat)
            
    if missing_descs:
        logger.error(f"âŒ CRITICAL: {len(missing_descs)} descriptor columns missing!")
        logger.error(f"First few missing: {missing_descs[:5]}")
        raise KeyError(f"Missing required descriptor column: {missing_descs[0]}")
    else:
        logger.info(f"âœ… All {len(plan.descriptor_names)} descriptor columns present.")

    logger.info("SCHEMA VALIDATION PASSED.")
    logger.info("="*50)

    return [c for c in available_columns if c in required_cols_set]


def build_feature_matrices(df: "pandas.DataFrame", plan: FeaturePlan) -> Tuple["numpy.ndarray", "numpy.ndarray"]:
    import numpy as np

    # Fingerprints: keep uint8 (as stored), later cast to float32 for model input.
    fp_block = df[plan.fp_input_columns].to_numpy(copy=False)
    fp_block = np.asarray(fp_block, dtype=np.uint8, order="C")

    # Descriptors: raw float32, keep NaN (no imputation).
    desc_df = df[plan.descriptor_names]
    try:
        desc_block = desc_df.to_numpy(dtype=np.float32, copy=False)
    except Exception:
        # Fallback: coerce column-wise to float (slower, but robust to bad dtypes).
        import pandas as pd

        desc_block = desc_df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32, copy=False)
    desc_block = np.asarray(desc_block, dtype=np.float32, order="C")
    return fp_block, desc_block


def apply_scaling_if_needed(
    fp_block: "numpy.ndarray",
    desc_block: "numpy.ndarray",
    scaler: Any,
    model_name: str,
) -> "numpy.ndarray":
    import numpy as np

    if model_name in NON_TREE_MODELS:
        if scaler is None:
            raise ValueError(f"Scaler is required for {model_name} but is None.")
        desc_scaled = scaler.transform(desc_block.astype(np.float32, copy=False))
        fp_f32 = fp_block.astype(np.float32, copy=False)
        return np.concatenate([fp_f32, np.asarray(desc_scaled, dtype=np.float32)], axis=1).astype(
            np.float32, copy=False
        )

    fp_f32 = fp_block.astype(np.float32, copy=False)
    return np.concatenate([fp_f32, desc_block.astype(np.float32, copy=False)], axis=1).astype(np.float32, copy=False)


def _norm_token(s: str) -> str:
    return "".join(ch.lower() for ch in str(s) if ch.isalnum() or ch in {"_", "-"}).replace("-", "_")


def resolve_threshold(row: Mapping[str, Any], mode: Union[str, float] = "auto", metric: str = "f1") -> float:
    """
    Resolve threshold value from a row in threshold_selection_summary.csv.

    Args:
        row: Dictionary-like row from threshold summary CSV
        mode: Either "auto" or a float threshold value
        metric: Metric to use when mode="auto" (f1, youden, or mcc)

    Returns:
        Float threshold in [0, 1]
    """
    # If manually specified, return that value
    if mode != "auto":
        try:
            return float(mode)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid threshold value: {mode}")

    # Auto-resolve based on metric
    if metric == "f1":
        for col in ["selected_threshold", "max_f1_threshold", "f1_threshold"]:
            if col in row and row[col] is not None and str(row[col]) != "nan":
                return float(row[col])

    elif metric == "youden":
        for col in ["youden_j_threshold", "youden_threshold", "j_threshold"]:
            if col in row and row[col] is not None and str(row[col]) != "nan":
                return float(row[col])

    elif metric == "mcc":
        for col in ["max_mcc_threshold", "mcc_threshold"]:
            if col in row and row[col] is not None and str(row[col]) != "nan":
                return float(row[col])
        # Fallback to youden if MCC not available
        import logging
        logger = logging.getLogger(__name__)
        logger.warning("MCC threshold not available, falling back to youden_j")
        for col in ["youden_j_threshold", "youden_threshold"]:
            if col in row and row[col] is not None and str(row[col]) != "nan":
                return float(row[col])

    else:
        raise ValueError(f"Unknown metric: {metric}")

    # If no column found for the requested metric, try fallback order
    for col in ["selected_threshold", "max_f1_threshold", "youden_j_threshold"]:
        if col in row and row[col] is not None and str(row[col]) != "nan":
            return float(row[col])

    return 0.5


def load_threshold_auto(
    paths: ArtifactPaths, model_name: str, seed: int, threshold_metric: str, logger: Any = None
) -> float:
    import pandas as pd

    if logger is None:
        import logging
        logger = logging.getLogger(__name__)

    if not paths.threshold_summary_path.exists():
        logger.warning(
            f"threshold_selection_summary.csv not found: {paths.threshold_summary_path} -> using 0.5"
        )
        return 0.5

    df = pd.read_csv(paths.threshold_summary_path)
    if "model" not in df.columns:
        raise ValueError(f"threshold_selection_summary.csv missing 'model' column: {paths.threshold_summary_path}")

    rows = df.copy()
    # Optional seed filtering if present.
    for seed_col in ["seed", "split_seed"]:
        if seed_col in rows.columns:
            rows = rows[pd.to_numeric(rows[seed_col], errors="coerce") == int(seed)]
            break

    # Model matching (exact or normalized token match).
    exact = rows[rows["model"].astype(str) == str(model_name)]
    rows = exact if not exact.empty else rows[rows["model"].astype(str).map(_norm_token) == _norm_token(model_name)]

    if rows.empty:
        raise ValueError(
            f"No threshold row found for model={model_name}, seed={seed} in {paths.threshold_summary_path}"
        )
    row = rows.iloc[0].to_dict()

    threshold = resolve_threshold(row=row, mode="auto", metric=threshold_metric)
    logger.info(f"Resolved threshold for metric '{threshold_metric}': {threshold:.6f}")
    return threshold

def compute_batch_ad(
    full_raw_features: np.ndarray,
    ad_artifacts: Dict[str, Any],
    plan: FeaturePlan,
) -> Dict[str, np.ndarray]:
    pca = ad_artifacts.get("pca")
    ad_scaler = ad_artifacts.get("scaler")
    train_pca_xtx_inv = ad_artifacts.get("train_pca_xtx_inv")
    train_fp_bin = ad_artifacts.get("train_fp_bin")
    train_desc_scaled = ad_artifacts.get("train_desc_scaled")
    weights = dict(ad_artifacts.get("weights", {}))
    if "ad_score_power" in ad_artifacts:
        weights.setdefault("ad_score_power", float(ad_artifacts["ad_score_power"]))

    n_samples = len(full_raw_features)
    if full_raw_features.shape[1] != plan.n_features_total:
        raise ValueError(
            f"AD input feature dim mismatch: {full_raw_features.shape[1]} vs expected {plan.n_features_total}"
        )

    n_fp = plan.n_fp
    lev = np.zeros(n_samples, dtype=np.float32)
    if pca is not None and ad_scaler is not None and train_pca_xtx_inv is not None:
        lev = _compute_leverage_pca(
            pca=pca,
            pca_scaler=ad_scaler,
            features=full_raw_features,
            train_pca_xtx_inv=train_pca_xtx_inv,
        )

    full_scaled = ad_scaler.transform(full_raw_features) if ad_scaler is not None else full_raw_features
    current_batch_fps = full_raw_features[:, :n_fp]
    current_batch_desc_scaled = full_scaled[:, n_fp:]

    max_tanimoto = np.zeros(n_samples, dtype=np.float32)
    if train_fp_bin is not None:
        max_tanimoto = _compute_tanimoto_similarity(train_fp_bin, current_batch_fps)

    max_cosine = np.zeros(n_samples, dtype=np.float32)
    if train_desc_scaled is not None:
        max_cosine = _compute_cosine_similarity(train_desc_scaled, current_batch_desc_scaled)

    density_score = _compute_density_score_from_reference(
        lev,
        reference_median=ad_artifacts.get("density_reference_median"),
    )
    ad_final = _compute_ad_score(density_score, max_tanimoto, max_cosine, weights)

    return {
        "AD_Score": ad_final,
        "leverage": lev,
        "max_tanimoto": max_tanimoto,
        "max_cosine": max_cosine,
    }

def predict_batch(
    df: "pd.DataFrame",
    plan: FeaturePlan,
    model: Any,
    scaler: Any,
    model_name: str,
    threshold: float,
    smiles_validation: str,
    ad_artifacts: Optional[Dict] = None,
) -> Tuple["pd.DataFrame", Dict[str, int]]:

    # 1. Perform basic input checks and filtering.
    n_in = len(df)
    if n_in == 0:
        return pd.DataFrame(), {"processed": 0, "predicted": 0, "skipped": 0}

    zinc_id = pd.to_numeric(df["zinc_id"], errors="coerce")
    smiles = _normalize_smiles_series(df["smiles"])
    valid_mask = zinc_id.notna() & smiles.notna() & (smiles.str.len() > 0)
    if valid_mask.sum() == 0:
        return pd.DataFrame(), {"processed": n_in, "predicted": 0, "skipped": n_in}

    dfv = df.loc[valid_mask].copy()
    
    # 2. Build fingerprint and descriptor matrices.
    fp_final, desc_final = build_feature_matrices(dfv, plan)
    nan_rows = np.isnan(desc_final).any(axis=1)
    
    # Optional SMILES validation.
    ok_smiles = np.ones((len(dfv),), dtype=bool)
    if smiles_validation == "rdkit":
        try:
            from rdkit import Chem
            smi_list = smiles.loc[valid_mask].to_numpy(dtype="object", copy=False).tolist()
            ok_smiles = np.fromiter((Chem.MolFromSmiles(str(s)) is not None for s in smi_list), 
                                   dtype=bool, count=len(smi_list))
        except ImportError: pass

    final_ok = (~nan_rows) & ok_smiles
    if final_ok.sum() == 0:
        return pd.DataFrame(), {"processed": n_in, "predicted": 0, "skipped": n_in}

    fp_calc = fp_final[final_ok]
    desc_calc = desc_final[final_ok]
    zid_calc = zinc_id.loc[valid_mask].to_numpy()[final_ok]
    smi_calc = smiles.loc[valid_mask].to_numpy()[final_ok]

    # 3. Run QSAR model inference.
    X = apply_scaling_if_needed(fp_calc, desc_calc, scaler=scaler, model_name=model_name)
    proba = model.predict_proba(X)[:, 1].astype(np.float32)

    # 4. Compute AD scores when AD artifacts are available.
    ad_score, lev, tanimoto, cosine = np.ones_like(proba), np.zeros_like(proba), np.zeros_like(proba), np.zeros_like(proba)

    if ad_artifacts:
        try:
            # Build the raw feature matrix required by the AD routine (FP + descriptors).
            full_raw_features = np.concatenate([
                fp_calc.astype(np.float32), 
                desc_calc.astype(np.float32)
            ], axis=1)

            # Call the AD scoring helper.
            ad_out = compute_batch_ad(
                full_raw_features=full_raw_features,
                ad_artifacts=ad_artifacts,
                plan=plan,
            )
            ad_score = ad_out["AD_Score"]
            lev = ad_out["leverage"]
            tanimoto = ad_out["max_tanimoto"]
            cosine = ad_out["max_cosine"]
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"AD inner error: {e}")

    # 5. Assemble the output table.
    out = pd.DataFrame({
        "zinc_id": zid_calc.astype(np.int64),
        "smiles": smi_calc,
        "prob": proba,
        "pred_label": (proba >= threshold).astype(np.int8),
        "AD_Score": ad_score,
        "leverage": lev,
        "max_tanimoto": tanimoto,
        "max_cosine": cosine
    })
    return out, {"processed": n_in, "predicted": len(out), "skipped": n_in - len(out)}

def stream_inference(
    input_path: Path,
    output_path: Path,
    plan: FeaturePlan,
    model: Any,
    scaler: Any,
    model_name: str,
    seed: int,
    threshold: float,
    batch_size: int,
    smiles_validation: str,
    ad_artifacts: Optional[Any] = None,
) -> None:
    import logging
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tqdm import tqdm
    
    logger = logging.getLogger(__name__)

    if ad_artifacts:
        logger.info(f"AD integration active: {list(ad_artifacts.keys())}")
    else:
        logger.warning("AD integration is INACTIVE (no artifacts provided)")

    pf = pq.ParquetFile(input_path)
    all_input_cols = pf.schema.names
    cols = select_required_input_columns(all_input_cols, plan)

    id_col = None
    for possible_name in ['zinc_id', 'id', 'ZINC_ID', 'compound_id']:
        if possible_name in all_input_cols:
            id_col = possible_name
            break
    if not id_col:
        raise ValueError(f"Could not find ID column in {input_path}.")
    
    logger.info(f"Detected ID column: '{id_col}', will map to 'zinc_id' in output.")

    total_rows = int(getattr(pf.metadata, "num_rows", 0) or 0)
    total_batches = (total_rows + batch_size - 1) // batch_size if total_rows > 0 else None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    out_schema = pa.schema([
        pa.field("zinc_id", pa.int64()),
        pa.field("smiles", pa.string()),
        pa.field("prob", pa.float32()),
        pa.field("pred_label", pa.int8()),
        pa.field("AD_Score", pa.float32()), 
        pa.field("leverage", pa.float32()),
        pa.field("max_tanimoto", pa.float32()), 
        pa.field("max_cosine", pa.float32()),   
        pa.field("model_name", pa.string()),
        pa.field("seed", pa.int64()),
        pa.field("threshold_used", pa.float32()),
    ])

    writer = pq.ParquetWriter(output_path, out_schema, compression="zstd")
    processed = predicted = skipped = 0

    pbar = tqdm(
        pf.iter_batches(batch_size=int(batch_size), columns=cols, use_threads=True),
        total=total_batches,
        desc="QSAR + AD Inference",
        unit="batch",
    )

    for batch_idx, batch in enumerate(pbar):
        try:
            df = batch.to_pandas()
            df = df.rename(columns={id_col: "zinc_id"})
            df["zinc_id"] = df["zinc_id"].astype("int64")
        except Exception as exc:
            logger.error(f"[Batch {batch_idx}] ERROR preparing batch: {exc}")
            continue

        try:
            out_df, stats = predict_batch(
                df=df,
                plan=plan,
                model=model,
                scaler=scaler,
                model_name=model_name,
                threshold=threshold,
                smiles_validation=smiles_validation,
                ad_artifacts=ad_artifacts  # Pass the full AD artifact bundle.
            )
        except Exception as exc:
            logger.error(f"[Batch {batch_idx}] ERROR during prediction/AD: {exc}")
            continue

        processed += int(stats["processed"])
        predicted += int(stats["predicted"])
        skipped += int(stats["skipped"])

        if len(out_df) == 0:
            continue

        out_df["model_name"] = str(model_name)
        out_df["seed"] = int(seed)
        out_df["threshold_used"] = np.float32(threshold)

        try:
            table = pa.Table.from_pandas(out_df, schema=out_schema, preserve_index=False)
            writer.write_table(table)
        except Exception as exc:
            logger.error(f"[Batch {batch_idx}] Schema mismatch: {exc}")

        if batch_idx % 20 == 0:
            pbar.set_postfix({"pred": predicted, "skip": skipped})

    writer.close()
    logger.info(f"✅ Success! Output saved to: {output_path}")


def sanity_check_first_batch(
    pf,
    cols: List[str],
    plan: FeaturePlan,
    expected_dim: int,
    sample_rows: int = 1000,
) -> None:
    import logging

    logger = logging.getLogger(__name__)
    it = pf.iter_batches(batch_size=int(sample_rows), columns=cols, use_threads=True)
    first = next(it, None)
    
    if first is None:
        raise RuntimeError("Input parquet appears empty; cannot run sanity check.")
    
    df = first.to_pandas()
    
    # Verify that an ID column is present.
    id_present = any(name in df.columns for name in ['zinc_id', 'id', 'ZINC_ID', 'compound_id'])
    if not id_present:
        raise ValueError(f"Missing ID column. Available: {df.columns.tolist()}")

    # Verify feature matrix dimensions.
    fp_u8, desc_f32 = build_feature_matrices(df, plan)
    
    logger.info("--- Pre-flight Sanity Check ---")
    logger.info(f"  ID found   : Yes")
    logger.info(f"  FP dim     : {fp_u8.shape[1]} (Expect: {plan.n_fp})")
    logger.info(f"  Desc dim   : {desc_f32.shape[1]} (Expect: {plan.n_desc})")
    logger.info(f"  Total dim  : {fp_u8.shape[1] + desc_f32.shape[1]} (Expect: {expected_dim})")
    
    if fp_u8.shape[1] != plan.n_fp or desc_f32.shape[1] != plan.n_desc:
        raise RuntimeError("Feature dimension mismatch! Check if zinc_features.parquet matches training config.")

    logger.info("--- Check Passed. Starting Full Inference ---")


def main(argv: Optional[Sequence[str]] = None) -> None:
    import logging
    
    _require_deps()
    import pyarrow.parquet as pq

    # Initialize logging with console output only (will add file handler after output_path is determined)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler()]
    )
    
    logger = logging.getLogger(__name__)

    args = parse_args(argv)
    model_dir: Path = args.model_dir
    model_name: str = str(args.model_name)
    seed: int = int(args.seed)
    calibration: str = str(args.calibration)
    batch_size: int = int(args.batch_size)
    input_path: Path = args.input

    if not input_path.exists():
        logger.error(f"Input parquet not found: {input_path}")
        raise FileNotFoundError(f"Input parquet not found: {input_path}")

    paths = build_artifact_paths(model_dir=model_dir, model_name=model_name, seed=seed, calibration=calibration)
    if not paths.feature_names_path.exists():
        raise FileNotFoundError(f"Missing feature_names_final.json: {paths.feature_names_path}")
    if not paths.descriptor_names_path.exists():
        raise FileNotFoundError(f"Missing descriptor_names.json: {paths.descriptor_names_path}")

    model, scaler, used_calibration = load_model(paths=paths, model_name=model_name, calibration=calibration)
    plan = load_feature_plan(paths=paths)
    validate_fp_mask(paths=paths, plan=plan)

    ad_artifacts = None
    if args.ad_integration:
        try:
            logger.info("Loading Applicability Domain artifacts...")
            ad_artifacts = load_ad_artifacts(paths.split_dir, model_name, seed, plan=plan)

            if ad_artifacts is None:
                logger.warning("AD artifacts loading returned None")
                ad_artifacts = {}
            else:
                logger.info("AD artifacts loaded successfully")
                if "pca" in ad_artifacts and ad_artifacts["pca"] is not None:
                    logger.info(f"  - PCA components: {ad_artifacts['pca'].n_components_}")
                if "train_fp_bin" in ad_artifacts:
                    logger.info(f"  - Training fingerprints: {ad_artifacts['train_fp_bin'].shape}")
                if "train_desc_scaled" in ad_artifacts:
                    logger.info(f"  - Training descriptors: {ad_artifacts['train_desc_scaled'].shape}")
                if "density_reference_median" in ad_artifacts:
                    logger.info(
                        f"  - Density reference median: {float(ad_artifacts['density_reference_median']):.6f}"
                    )
        except Exception as e:
            logger.error(f"Failed to load AD artifacts: {e}")
            logger.error("Continuing without AD integration")
            ad_artifacts = None

    if str(args.threshold).strip().lower() == "auto":
        threshold = load_threshold_auto(
            paths=paths,
            model_name=model_name,
            seed=seed,
            threshold_metric=str(args.threshold_metric),
            logger=logger,
        )
    else:
        threshold = _safe_float(args.threshold)
    if not (0.0 <= float(threshold) <= 1.0):
        raise ValueError(f"Threshold must be within [0, 1], got {threshold}")

    run_dir = paths.run_dir
    output_path: Path
    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = run_dir / "virtual_screening" / f"zinc_predictions_{ts}.parquet"
    else:
        output_path = args.output

    # Add file handler to logger after output_path is determined
    log_dir = output_path.parent
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(file_handler)
    logger.info(f"Starting QSAR virtual screening inference. Log file: {log_path}")

    # Quick schema sanity check (fail fast before streaming)
    pf = pq.ParquetFile(input_path)
    cols = select_required_input_columns(pf.schema.names, plan)

    expected_dim = len(plan.feature_names_final)
    sanity_check_first_batch(pf=pf, cols=cols, plan=plan, expected_dim=expected_dim, sample_rows=1000)

    logger.info("Virtual screening inference configuration:")
    logger.info(f"  Run dir    : {run_dir}")
    logger.info(f"  Split dir  : {paths.split_dir}")
    logger.info(f"  Input      : {input_path}")
    logger.info(f"  Output     : {output_path}")
    logger.info(f"  Model      : {model_name} (requested_calibration={calibration}, used={used_calibration})")
    logger.info(f"  Seed       : {seed}")
    logger.info(f"  Features   : n_fp={plan.n_fp} n_desc={plan.n_desc} total={len(plan.feature_names_final)}")
    logger.info(f"  Threshold  : {threshold:.6f} ({'auto' if str(args.threshold).strip().lower()=='auto' else 'manual'})")
    logger.info(f"  Batch size : {batch_size:,}")
    logger.info(f"  SMILES val : {str(args.smiles_validation)}")

    stream_inference(
        input_path=input_path,
        output_path=output_path,
        plan=plan,
        model=model,
        scaler=scaler,
        model_name=model_name,
        seed=seed,
        threshold=float(threshold),
        batch_size=batch_size,
        smiles_validation=str(args.smiles_validation),
        ad_artifacts=ad_artifacts
    )


if __name__ == "__main__":
    # Avoid tokenizers/BLAS oversubscription on big batches
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()
