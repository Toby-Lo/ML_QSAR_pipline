#!/usr/bin/env python3
"""
Applicability Domain (AD) analysis for step10_qsar_ml.py outputs.

This script is adapted to the artifact layout produced by:
  ./scripts/step10_qsar_ml.py

It follows the core logic of the reference implementation:
  - Leverage (PCA-based) + Williams plot statistics
  - Optional SOM occupancy (MiniSom)
  - Similarity filters: max Tanimoto (fingerprints) + max cosine (scaled full features)

The script is organized with ``# %%`` blocks:
  - Upper section: compute + export data (recommended)
  - Lower section: optional plotting cells (commented out by default)

Usage (CLI):
[Auto]
  python scripts/step22_applicability_domain.py \
    --run-dir models_out/qsar_ml_20260412_162829 \
    --split-seed 12345  --model SVC \
    --learn-weights \
    --base-method leverage \
    --tanimoto-threshold 0.80 \
    --cosine-threshold 0.80 \
    --strict-similarity \
    --base-feature-space full \
    --logit-shrinkage-method auto \
    --leverage-pca-variance 0.95 \
    --compare-calibration

[manual]
    python scripts/step22_applicability_domain.py \
    --run-dir models_out/qsar_ml_20260412_162829 \
    --split-seed 12345 --model SVC \
    --base-feature-space full \
    --ad-score-power 2.0 \
    --w1-tanimoto 0.9  --w2-cosine 0.1  --w3-similarity 0.2  --w4-density 0.8 \
    --base-method leverage \
    --leverage-pca-variance 0.95 \
    --tanimoto-threshold 0.80 \
    --domain-threshold-quantile 0.90 \
    --strict-similarity \
    --compare-calibration

[final]
  python scripts/step22_applicability_domain.py\
    --run-dir models_out/qsar_ml_20260412_162829 \
    --split-seed 12345 --model SVC \
    --base-method leverage  --leverage-pca-components 70 \
    --base-feature-space full \
    --w1-tanimoto 0.9  --w2-cosine 0.1  --w3-similarity 0.05  --w4-density 0.95 \
    --ad-score-power 2.0 \
    --tanimoto-threshold 0.85  --cosine-threshold 0.70 \
    --strict-similarity \
    --logit-shrinkage-method auto \
    --compare-calibration

Usage (interactive):
  1) Run the compute/export block (CLI or Jupyter cell)
  2) Then uncomment the plotting cells below to iterate on visualizations without recomputing
"""

# %%
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Ensure project root is in Python path for utils imports
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# %%
# Heavy deps are imported here so the upper section can be edited easily.
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr

try:
    from minisom import MiniSom  # type: ignore
except Exception:
    MiniSom = None  # type: ignore

# Import calibration integration module from utils
import logging
try:
    from utils.calibration_integration import (
        try_load_calibrated_predictions,
        apply_calibrated_model_to_external,
        flexible_column_resolution,
        compute_calibration_comparison_curve,
        apply_ad_shrinkage,
        integrate_calibration_to_ad_config,
    )
    _CALIBRATION_INTEGRATION_AVAILABLE = True
except ImportError as e:
    _CALIBRATION_INTEGRATION_AVAILABLE = False
    logging.debug(f"Calibration integration not available from utils.calibration_integration: {e}")
    # Graceful fallback
    try_load_calibrated_predictions = None
    apply_calibrated_model_to_external = None
    flexible_column_resolution = None
    compute_calibration_comparison_curve = None
    apply_ad_shrinkage = None
    integrate_calibration_to_ad_config = None


# %%
@dataclass
class ADConfig:
    run_dir: Path
    split_seed: int
    model_key: str

    train_npz: Optional[Path] = None
    external_npz: Optional[Path] = None
    predictions_csv: Optional[Path] = None

    id_column: str = "id"
    smiles_column: str = "smiles"
    y_true_column: str = "y_true"
    y_prob_column: str = "y_prob"
    model_column: str = "model"

    # Leverage / Williams
    leverage_pca_variance: float = 0.95
    leverage_pca_components: Optional[int] = None  # If set, overrides variance_ratio (fixed n_components)
    williams_residual_z: float = 3.0

    # Base AD method (applied in a chosen feature space)
    # - "leverage": PCA leverage (Williams-style)
    # - "mahalanobis": robust Mahalanobis distance (Ledoit-Wolf shrinkage) with train-quantile threshold
    # - "knn_density": mean distance to kNN with train-quantile threshold (cheap SOM alternative)
    base_method: str = "knn_density"
    base_feature_space: str = "full"  # "desc" | "fingerprint" | "full"
    domain_threshold_quantile: float = 0.99
    knn_k: int = 5

    # Similarity thresholds
    tanimoto_threshold: float = 0.60
    cosine_threshold: float = 0.70
    # If True: in_domain = (leverage & som) AND (tanimoto & cosine)
    # If False: in_domain = (leverage & som) OR  (tanimoto & cosine)
    strict_similarity: bool = True

    # SOM
    skip_som: bool = True
    som_rows: int = 12
    som_cols: int = 12
    som_iterations: int = 5000

    # Output
    output_dir: Optional[Path] = None
    inplace_update_predictions: bool = False
    make_plots: bool = False

    # Performance
    cosine_block_size: int = 1024

    # Weight learning (TASK 1)
    learn_weights: bool = False  # Learn weights from dev set
    calibration_method: str = "isotonic"  # isotonic | sigmoid
    weight_search_grid: int = 20  # Grid resolution (0.0, 0.05, 0.10, ..., 1.0)

    # Manual weight configuration (used when learn_weights=False)
    w1_tanimoto: Optional[float] = None  # Tanimoto weight (0.0-1.0), None uses default 0.7
    w2_cosine: Optional[float] = None    # Cosine weight (0.0-1.0), None uses default 0.3
    w3_similarity: Optional[float] = None  # Similarity weight (0.0-1.0), None uses default 0.6
    w4_density: Optional[float] = None    # Density weight (0.0-1.0), None uses default 0.4

    # AD weight config (can be overridden, takes precedence over individual w1-w4 values)
    ad_weight_config: Optional[Dict[str, float]] = None

    # AD Score power for non-linear shrinkage
    ad_score_power: float = 2.0
    """
    Power exponent for AD score: Final_Score = Prob * (AD_Score)^k
    k=1.0: linear (default old behavior would be 1.0)
    k=2.0: quadratic (strong penalty for low AD scores)
    k=3.0: cubic (even stronger penalty)
    Higher k values suppress predictions with low applicability domain scores.
    """

    #  Calibration integration configuration
    logit_shrinkage_method: str = "auto"
    """
    Apply AD shrinkage strategy:
    - "auto": Detect calibration; use probability_space if found, else conservative
    - "probability_space": p * AD_Score (preserves calibration properties)
    - "logit_space": Apply in logit space (may damage calibration)
    - "conservative": Mixed strategy, minimal loss
    - "none": No shrinkage, use AD_Score directly
    """

    detect_calibration: bool = True
    """Auto-detect and apply step20 calibration results"""

    compare_pre_post_calibration: bool = True
    """Generate AD effectiveness comparison curves (pre vs post calibration)"""

    # Metadata filled by auto-detection
    calibration_available: bool = False
    calibration_metadata: Optional[Dict[str, Any]] = None
    recommended_shrinkage_method: str = "conservative"


# %%
"""
Config cell (edit this first)
-----------------------------

Defaults match the requested CLI-like settings:
  --tanimoto-threshold 0.70
  --cosine-threshold 0.80
  --no-strict-similarity  (i.e., strict_similarity=False => leverage OR similarity)
"""

USER_CONFIG: Dict[str, Any] = {
    # If None => auto-pick latest under models_out/qsar_ml_*
    "run_dir": None,
    "split_seed": 42,
    "model_key": "SVC",

    "tanimoto_threshold": 0.80,
    "cosine_threshold": 0.80,
    "strict_similarity": True,

    # Base domain method
    "base_method": "knn_density",  # leverage | mahalanobis | knn_density
    "base_feature_space": "desc",  # desc | full
    "domain_threshold_quantile": 0.99,
    "knn_k": 5,

    # Optional SOM (requires minisom)
    "skip_som": True,
    "som_rows": 12,
    "som_cols": 12,
    "som_iterations": 5000,

    "leverage_pca_variance": 0.95,
    "williams_residual_z": 3.0,
    "cosine_block_size": 1024,

    # Outputs
    "output_dir": None,
    "inplace_update_predictions": False,
    "make_plots": False,
}


# %%
def _resolve_latest_run_dir(models_out: Path = Path("models_out")) -> Path:
    candidates = sorted([p for p in models_out.glob("qsar_ml_*") if p.is_dir()])
    if not candidates:
        raise FileNotFoundError("No run folder found under models_out (pattern: qsar_ml_*)")
    return candidates[-1]


def _default_paths(run_dir: Path, split_seed: int) -> Tuple[Path, Path, Path]:
    split_dir = run_dir / f"split_seed_{split_seed}"
    train_npz = split_dir / "data" / "splits" / "dev_train.npz"
    external_npz = split_dir / "data" / "splits" / "external_test.npz"
    predictions_csv = split_dir / "predictions" / "external_test_predictions.csv"
    return train_npz, external_npz, predictions_csv


def _load_npz(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"NPZ not found: {path}")
    with np.load(path, allow_pickle=True) as data:
        required = {"fp", "desc", "y", "id", "smiles"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"NPZ missing keys {sorted(missing)}: {path}")
        return {k: data[k] for k in data.files}


def _binarize_fp(fp: np.ndarray) -> np.ndarray:
    return np.clip(np.round(fp).astype(np.int8), 0, 1)


def _tanimoto_max(train_fp_bin: np.ndarray, query_fp_bin: np.ndarray) -> np.ndarray:
    """Return max Tanimoto similarity of each query to the training set."""
    train = train_fp_bin.astype(np.int32, copy=False)
    queries = query_fp_bin.astype(np.int32, copy=False)
    train_sum = train.sum(axis=1).astype(np.int32)
    out = np.zeros(len(queries), dtype=np.float32)
    for i, q in enumerate(queries):
        inter = train @ q
        union = train_sum + int(q.sum()) - inter
        union = np.where(union == 0, 1, union)
        sims = inter / union
        out[i] = float(np.max(sims)) if sims.size else 0.0
    return out


def _cosine_max_blockwise(X_query: np.ndarray, X_train: np.ndarray, block_size: int = 1024) -> np.ndarray:
    """Return max cosine similarity of each query to X_train (blockwise to reduce peak memory)."""
    eps = 1e-12
    train = X_train.astype(np.float32, copy=False)
    query = X_query.astype(np.float32, copy=False)

    train_norm = np.linalg.norm(train, axis=1, keepdims=True)
    train_norm = np.where(train_norm < eps, 1.0, train_norm)
    train_unit = train / train_norm

    out = np.zeros(len(query), dtype=np.float32)
    for start in range(0, len(query), int(block_size)):
        end = min(len(query), start + int(block_size))
        q = query[start:end]
        q_norm = np.linalg.norm(q, axis=1, keepdims=True)
        q_norm = np.where(q_norm < eps, 1.0, q_norm)
        q_unit = q / q_norm
        sims = q_unit @ train_unit.T
        out[start:end] = np.max(sims, axis=1).astype(np.float32)
    return out


def _compute_leverage_pca(
    X_train_scaled: np.ndarray,
    X_query_scaled: np.ndarray,
    variance_ratio: float,
    fixed_components: Optional[int] = None,
    ad_output_dir: Optional[Path] = None,
    scaler_to_save: Optional[Any] = None
) -> Tuple[np.ndarray, float, int]:
    """
    Compute leverage in a PCA subspace.
    
    Args:
        X_train_scaled: Training feature matrix (standardized)
        X_query_scaled: Query feature matrix (standardized)
        variance_ratio: PCA variance ratio (0.0-1.0) if fixed_components is None
        fixed_components: If specified (int), use fixed n_components instead of variance ratio
    
    Returns:
        (leverage_scores, h_star_threshold, n_components)
    """

    # Debug
    import numpy as np
    import joblib
    from sklearn.decomposition import PCA

    train_var = np.var(X_train_scaled, axis=0)
    zero_var_cols = np.sum(train_var < 1e-9)
    logging.info(f"[PCA] 输入形状: {X_train_scaled.shape}, 零方差列数: {zero_var_cols}")
    
    if X_train_scaled.shape[1] - zero_var_cols < 1:
         raise ValueError("错误：没有有效的特征可以用于 PCA 计算！")

    # --- 2. 配置 PCA 策略 ---
    if fixed_components is not None:
        n_comp = int(max(1, min(fixed_components, X_train_scaled.shape[1])))
        pca = PCA(n_components=n_comp, svd_solver="full")
        logging.debug(f"使用固定主成分数: {n_comp}")
    else:
        # 自动根据方差贡献率选择
        pca = PCA(n_components=float(variance_ratio), svd_solver="full")
        logging.debug(f"使用方差贡献率阈值: {variance_ratio:.2%}")
    
    # --- 3. 执行拟合与投影 ---
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_query_pca = pca.transform(X_query_scaled)
    n_p = int(X_train_pca.shape[1])
    logging.info(f"PCA 提取完成，实际保留成分数: {n_p}")

    # --- 4. 关键：导出模型与配套 Scaler ---
    if ad_output_dir:
        ad_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存 PCA 模型
        pca_path = ad_output_dir / "ad_pca_model.joblib"
        joblib.dump(pca, pca_path)
        logging.info(f"已导出 AD PCA 模型: {pca_path}")
        
        # 保存配套的 Scaler (如果没有它，推理时的特征缩放会不一致)
        if scaler_to_save:
            scaler_path = ad_output_dir / "ad_pca_scaler.joblib"
            joblib.dump(scaler_to_save, scaler_path)
            logging.info(f"已导出配套 Scaler: {scaler_path}")

    # --- 5. 计算 Leverage (Hat Matrix 对角线) ---
    # 公式: h = diag(X_q @ (X_train^T @ X_train)^-1 @ X_q^T)
    try:
        XtX = X_train_pca.T @ X_train_pca
        inv = np.linalg.pinv(XtX) # 使用伪逆确保数值稳定性
        h_query = np.einsum("ij,jk,ik->i", X_query_pca, inv, X_query_pca)
    except np.linalg.LinAlgError as e:
        logging.error(f"矩阵求逆失败: {e}")
        h_query = np.zeros(X_query_pca.shape[0])

    # 计算警告阈值 h* (3p/n)
    n_samples = int(X_train_pca.shape[0])
    h_star = (3.0 * n_p) / n_samples if n_samples > 0 else 0.0
    
    return h_query.astype(np.float64), float(h_star), int(X_train_pca.shape[1])


def _per_sample_log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Per-sample log loss (cross-entropy) for binary classification."""
    p = np.clip(y_prob.astype(np.float64), eps, 1.0 - eps)
    y = y_true.astype(np.float64)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def _deviance_residual(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    """Signed deviance residual for binary classification (a better 'residual' than y_true - y_prob)."""
    ll = _per_sample_log_loss(y_true, y_prob)
    signed = np.sign(y_true.astype(np.float64) - y_prob.astype(np.float64))
    signed = np.where(signed == 0, 1.0, signed)
    return signed * np.sqrt(2.0 * ll)


def _zscore(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    sigma = float(np.std(x, ddof=1))
    if sigma == 0.0 or np.isnan(sigma):
        sigma = 1.0
    mu = float(np.mean(x))
    return (x - mu) / sigma


def _train_som(X_train_scaled: np.ndarray, rows: int, cols: int, iterations: int) -> Tuple[Any, set]:
    if MiniSom is None:
        raise SystemExit("MiniSom is required for SOM AD analysis. Install with `pip install minisom`.")
    som = MiniSom(int(rows), int(cols), X_train_scaled.shape[1], sigma=1.0, learning_rate=0.5)
    som.random_weights_init(X_train_scaled)
    som.train_random(X_train_scaled, int(iterations))
    occupied = {som.winner(x) for x in X_train_scaled}
    return som, occupied


def _som_flags(som: Any, occupied: set, X_query_scaled: np.ndarray) -> np.ndarray:
    flags: List[bool] = []
    for x in X_query_scaled:
        flags.append(som.winner(x) in occupied)
    return np.asarray(flags, dtype=bool)


def _align_predictions_to_external(
    predictions_csv: Path,
    external_ids: np.ndarray,
    external_smiles: np.ndarray,
    model_key: str,
    *,
    id_column: str,
    smiles_column: str,
    model_column: str,
) -> pd.DataFrame:
    preds = pd.read_csv(predictions_csv)
    if model_column not in preds.columns:
        raise ValueError(f"Predictions missing '{model_column}' column: {predictions_csv}")

    sub = preds[preds[model_column].astype(str).str.upper() == str(model_key).upper()].copy()
    if sub.empty:
        raise ValueError(f"No rows for model={model_key} in {predictions_csv}")

    if id_column not in sub.columns:
        raise ValueError(f"Predictions missing '{id_column}' column: {predictions_csv}")

    ext_order = pd.DataFrame(
        {
            id_column: external_ids.astype(str),
            smiles_column: external_smiles.astype(str) if smiles_column else external_smiles.astype(str),
        }
    )

    # Prefer (id,smiles) join when smiles exists on both sides; fallback to id-only.
    merged = None
    if smiles_column and smiles_column in sub.columns and smiles_column in ext_order.columns:
        try:
            merged = ext_order.merge(sub, on=[id_column, smiles_column], how="left", validate="one_to_one")
        except Exception:
            merged = None
    if merged is None or merged.isna().any().any():
        merged = ext_order[[id_column]].merge(sub, on=[id_column], how="left")

    # Strict 1:1 mapping validation
    if len(merged) != len(ext_order):
        raise ValueError(
            f"Row count mismatch after merge: external={len(ext_order)}, matched={len(merged)}. "
            f"Sample of missing IDs: {set(ext_order[id_column]) - set(merged[id_column])}"
        )
    if merged.isna().any().any():
        missing = merged[merged.isna().any(axis=1)]
        raise ValueError(
            "Failed to align predictions to external set. "
            f"Missing rows after merge: {len(missing)}/{len(merged)}. "
            f"Missing IDs: {missing[id_column].tolist()[:5]}"
        )
    return merged


def _backup_file(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        backup.write_bytes(path.read_bytes())


def _quantile_threshold(values: np.ndarray, q: float) -> float:
    q = float(q)
    q = min(max(q, 0.0), 1.0)
    return float(np.quantile(values.astype(np.float64), q))


def _mahalanobis_distance(train_scaled: np.ndarray, query_scaled: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (train_dist, query_dist) using Ledoit-Wolf covariance shrinkage."""
    estimator = LedoitWolf().fit(train_scaled)
    cov_inv = estimator.precision_
    center = estimator.location_
    train_centered = train_scaled - center
    query_centered = query_scaled - center
    d_train = np.einsum("ij,jk,ik->i", train_centered, cov_inv, train_centered)
    d_query = np.einsum("ij,jk,ik->i", query_centered, cov_inv, query_centered)
    return np.sqrt(np.maximum(d_train, 0.0)), np.sqrt(np.maximum(d_query, 0.0))


def _knn_mean_distance(train_scaled: np.ndarray, query_scaled: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (train_mean_dist, query_mean_dist) using Euclidean distances to kNN."""
    k = int(max(1, k))
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(train_scaled)), metric="euclidean")
    nn.fit(train_scaled)
    # For training samples: exclude self-match by using k+1 then dropping the first.
    tr_dist, _ = nn.kneighbors(train_scaled, return_distance=True)
    if tr_dist.shape[1] > 1:
        tr_mean = np.mean(tr_dist[:, 1:], axis=1)
    else:
        tr_mean = tr_dist[:, 0]
    te_dist, _ = nn.kneighbors(query_scaled, n_neighbors=min(k, len(train_scaled)), return_distance=True)
    te_mean = np.mean(te_dist, axis=1) if te_dist.size else np.zeros(len(query_scaled), dtype=float)
    return tr_mean.astype(np.float64), te_mean.astype(np.float64)


# %%
def _logit_transform(p: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Transform probability to logit space: log(p / (1-p))."""
    p = np.clip(p.astype(np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _sigmoid_transform(logit: np.ndarray) -> np.ndarray:
    """Transform logit back to probability space: 1 / (1 + exp(-logit))."""
    logit = np.asarray(logit, dtype=np.float64)
    # Clip to avoid overflow
    logit_clipped = np.clip(logit, -500, 500)
    return 1.0 / (1.0 + np.exp(-logit_clipped))


def _compute_density_score_unified(
    density_arr: np.ndarray,
    base_method: str,
    reference_arr: Optional[np.ndarray] = None,
    reference_median: Optional[float] = None,
    eps: float = 1e-9
) -> np.ndarray:
    """
    TASK 2: Unify density computation across all base methods.
    
    Apply unified median-decay normalization function:
        Score = 1 / (1 + d/d_0)
    
    where:
        - d = distance from query to training set (method-dependent)
        - d_0 = median distance in training set
        - Score ∈ [0, 1]: higher distance ⟹ lower score
    
    This ensures consistent normalized scaling regardless of which base_method
    produced the raw distances.
    
    Mathematical Properties:
        - At d = 0: Score = 1 (closest to training set)
        - At d = d_0: Score = 0.5 (at median distance)
        - At d → ∞: Score → 0
        - Monotonically decreasing, smooth
    
    Args:
        density_arr: Distance/leverage values (shape: N,), raw output from base method
        base_method: One of ["knn_density", "mahalanobis", "leverage"]
        reference_arr: Optional training-set reference array used to define a stable scale.
        reference_median: Optional explicit median reference. Overrides reference_arr if provided.
        eps: Epsilon for numerical stability (prevents division by zero)
    
    Returns:
        density_score: Normalized scores in [0, 1] (shape: N,)
    """
    density_arr = np.asarray(density_arr, dtype=np.float64)
    
    # Clean: remove negative/NaN/inf values (set to 0 distance)
    density_arr = np.where(
        (np.isnan(density_arr) | np.isinf(density_arr) | (density_arr < 0)),
        0.0,
        density_arr
    )
    
    ref = density_arr if reference_arr is None else np.asarray(reference_arr, dtype=np.float64)
    ref = np.where(
        (np.isnan(ref) | np.isinf(ref) | (ref < 0)),
        0.0,
        ref
    )

    # Compute median as reference point (robust measure of typical distance).
    # Using a training-set reference keeps AD scores stable across different query batches.
    d0 = float(reference_median) if reference_median is not None else float(np.median(ref))
    d0 = max(d0, eps)  # Numerical stability: avoid division by zero
    
    # Apply median-decay function: Score = 1 / (1 + d/d_0)
    # This normalizes distances in a way that's independent of scale
    density_score = 1.0 / (1.0 + (density_arr / d0))
    
    # Strictly bound to [0, 1] (mathematical guarantee)
    density_score = np.clip(density_score, 0.0, 1.0)
    
    # Verify bounds
    assert np.all((density_score >= 0.0) & (density_score <= 1.0)), \
        f"Density score violated bounds [0, 1]: min={np.min(density_score)}, max={np.max(density_score)}"
    
    logging.debug(
        f"Unified density normalization ({base_method}): "
        f"d_0={d0:.4e}, Score μ={np.mean(density_score):.3f}, "
        f"σ={np.std(density_score):.3f}, range=[{np.min(density_score):.3f}, {np.max(density_score):.3f}]"
    )
    
    return density_score


def _optimize_ad_weights(
    tanimoto: np.ndarray,
    cosine: np.ndarray,
    density: np.ndarray,
    error: np.ndarray,
    grid_resolution: int = 20,
    eps: float = 1e-9
) -> Dict[str, float]:
    """
    TASK 1: Learn optimal AD weights from dev set using grid search.
    
    Optimize:
        Sim_Score = w1 * Tanimoto + w2 * Cosine (w1 + w2 = 1)
        AD_Score = w3 * Sim_Score + w4 * Density (w3 + w4 = 1)
    
    Objective: Maximize negative SPEARMAN correlation between AD_Score and error.
    Higher AD_Score should correlate with lower error.
    
    Mathematical Consistency:
        Spearman correlation is rank-based, more robust to outliers than Pearson.
        We seek: AD_Score ↑ ⟹ Error ↓  (negative correlation is desired)
        Thus we maximize (−ρ) to get the most anti-correlated weights.
    
    Returns:
        {w1, w2, w3, w4, correlation} where:
            - w1 + w2 = 1 (Tanimoto + Cosine in similarity score)
            - w3 + w4 = 1 (Similarity + Density in AD score)
            - correlation: the achieved Spearman correlation (should be negative)
    """
    from scipy.stats import spearmanr
    
    # Normalize inputs to [0, 1] with epsilon guards
    tanimoto = np.clip(tanimoto.astype(np.float64), 0.0, 1.0)
    cosine = np.clip(cosine.astype(np.float64), 0.0, 1.0)
    density = np.clip(density.astype(np.float64), 0.0, 1.0)
    error = np.asarray(error, dtype=np.float64)

    # Remove NaNs and infinities
    mask = (
        ~(np.isnan(tanimoto) | np.isinf(tanimoto)) &
        ~(np.isnan(cosine) | np.isinf(cosine)) &
        ~(np.isnan(density) | np.isinf(density)) &
        ~(np.isnan(error) | np.isinf(error))
    )
    
    if not np.any(mask):
        logging.warning("No valid data for weight optimization; returning defaults")
        return {
            "w1_tanimoto": 0.5,
            "w2_cosine": 0.5,
            "w3_similarity": 0.5,
            "w4_density": 0.5,
            "correlation": 0.0,
            "n_samples": 0,
        }

    tanimoto = tanimoto[mask]
    cosine = cosine[mask]
    density = density[mask]
    error = error[mask]
    
    n_samples = len(error)
    logging.info(f"Optimizing weights on {n_samples} valid samples")

    # Guard against constant arrays (correlation undefined)
    std_checks = [
        (np.std(error), "error"),
        (np.std(density), "density"),
        (np.std(tanimoto), "tanimoto"),
        (np.std(cosine), "cosine"),
    ]
    
    for std_val, name in std_checks:
        if std_val < eps:
            logging.warning(f"Constant array detected for {name}; returning defaults")
            return {
                "w1_tanimoto": 0.5,
                "w2_cosine": 0.5,
                "w3_similarity": 0.5,
                "w4_density": 0.5,
                "correlation": 0.0,
                "n_samples": n_samples,
            }

    # Create grid of weight values
    w_values = np.linspace(0.0, 1.0, grid_resolution + 1)

    best_corr = 0.0  # Best (most negative) correlation
    best_w1, best_w3 = 0.5, 0.5
    best_config = None

    # Grid search over all combinations
    for w1 in w_values:
        w2 = 1.0 - w1
        sim_score = w1 * tanimoto + w2 * cosine
        
        # Guard sim_score
        if np.std(sim_score) < eps:
            continue

        for w3 in w_values:
            w4 = 1.0 - w3
            ad_score = w3 * sim_score + w4 * density
            
            # Guard ad_score
            if np.std(ad_score) < eps:
                continue

            # Compute SPEARMAN correlation (rank-based, robust)
            try:
                corr, pval = spearmanr(ad_score, error)
                if np.isnan(corr) or np.isinf(corr):
                    corr = 0.0
            except Exception as e:
                logging.debug(f"Correlation computation failed: {e}")
                corr = 0.0

            # Objective: maximize negative correlation (i.e., AD↑ ⟹ Error↓)
            # We want the most negative correlation, so we check if it's lower (more negative)
            if corr < best_corr:
                best_corr = corr
                best_w1 = w1
                best_w3 = w3
                best_config = {
                    "w1": w1,
                    "w2": w2,
                    "w3": w3,
                    "w4": w4,
                    "corr": corr,
                }

    # Ensure weights are stable and sum to 1
    if best_config is None:
        logging.warning("No valid configuration found in grid search; using defaults")
        return {
            "w1_tanimoto": 0.5,
            "w2_cosine": 0.5,
            "w3_similarity": 0.5,
            "w4_density": 0.5,
            "correlation": 0.0,
            "n_samples": n_samples,
        }

    w1 = float(np.clip(best_config["w1"], 0.0, 1.0))
    w2 = float(np.clip(best_config["w2"], 0.0, 1.0))
    w3 = float(np.clip(best_config["w3"], 0.0, 1.0))
    w4 = float(np.clip(best_config["w4"], 0.0, 1.0))
    
    logging.info(
        f"✓ Weight optimization complete: "
        f"w1={w1:.3f}, w2={w2:.3f}, w3={w3:.3f}, w4={w4:.3f}, "
        f"Spearman_corr={best_corr:.4f}"
    )
    
    return {
        "w1_tanimoto": w1,
        "w2_cosine": w2,
        "w3_similarity": w3,
        "w4_density": w4,
        "correlation": float(best_corr),
        "correlation_type": "spearman",
        "n_samples": n_samples,
        "optimization_method": "grid_search",
    }


def _compute_calibration_curve(
    ad_score: np.ndarray,
    error: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-9
) -> pd.DataFrame:
    """
    TASK 4: Compute calibration curve (AD_Score bins vs mean error).
    
    TASK 5: Engineering stability with epsilon guards.
    
    Returns:
        DataFrame with columns: [ad_min, ad_max, ad_mean, error_mean, error_std, count]
    """
    ad_score = np.asarray(ad_score, dtype=np.float64)
    error = np.asarray(error, dtype=np.float64)
    
    # Guard against NaN/inf
    ad_score = np.where((np.isnan(ad_score) | np.isinf(ad_score)), 0.5, ad_score)
    error = np.where((np.isnan(error) | np.isinf(error)), np.median(error[~np.isnan(error)]), error)
    
    results = []
    for i in range(n_bins):
        ad_min = i / n_bins
        ad_max = (i + 1) / n_bins
        
        mask = (ad_score >= ad_min) & (ad_score < ad_max)
        if np.any(mask):
            error_bin = error[mask]
            error_mean = float(np.mean(error_bin)) if len(error_bin) > 0 else 0.0
            error_std = float(np.std(error_bin)) if len(error_bin) > 1 else 0.0
            results.append({
                "ad_min": float(ad_min),
                "ad_max": float(ad_max),
                "ad_mean": float(np.mean(ad_score[mask])),
                "error_mean": error_mean,
                "error_std": error_std,
                "count": int(np.sum(mask)),
            })
    
    return pd.DataFrame(results)


def _compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    TASK 4: Compute Brier Score.
    
    Brier Score = mean((y_prob - y_true)^2)
    Measures calibration quality: lower is better (0 = perfect, 1 = worst)
    
    Args:
        y_true: True binary labels (0 or 1)
        y_prob: Predicted probabilities [0, 1]
    
    Returns:
        Brier score (float)
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.clip(np.asarray(y_prob, dtype=np.float64), 0.0, 1.0)
    
    if len(y_true) == 0:
        return 0.0
    
    brier = np.mean((y_prob - y_true) ** 2)
    return float(brier)


def _compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-9
) -> float:
    """
    TASK 4: Compute Expected Calibration Error (ECE).
    
    ECE = sum_b ( |pred_b - freq_b| * n_b / N )
    
    where:
        - b: bin index
        - pred_b: average predicted probability in bin b
        - freq_b: fraction of positives in bin b
        - n_b: number of samples in bin b
        - N: total samples
    
    Measures whether predicted probabilities match observed frequencies.
    Lower ECE = better calibration.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for discretization
        eps: Epsilon for numerical stability
    
    Returns:
        ECE (float) ∈ [0, 1]
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.clip(np.asarray(y_prob, dtype=np.float64), eps, 1.0 - eps)
    
    if len(y_true) == 0:
        return 0.0
    
    ece = 0.0
    n_total = len(y_true)
    
    for i in range(n_bins):
        prob_min = i / n_bins
        prob_max = (i + 1) / n_bins
        
        # Select samples in this probability range
        mask = (y_prob >= prob_min) & (y_prob < prob_max)
        
        if np.sum(mask) == 0:
            continue
        
        n_bin = np.sum(mask)
        # Predicted probability: mean of predictions in bin
        pred_prob = np.mean(y_prob[mask])
        # Empirical frequency: fraction of positives in bin
        freq_prob = np.mean(y_true[mask])
        
        # Contribution to ECE weighted by bin size
        ece += np.abs(pred_prob - freq_prob) * (n_bin / n_total)
    
    return float(ece)


def _compute_calibration_diagnostics(
    y_prob_raw: np.ndarray,
    y_prob_calibrated: Optional[np.ndarray],
    y_true: np.ndarray,
    ad_score: np.ndarray,
    eps: float = 1e-9,
) -> Dict[str, Any]:
    """
    TASK 4: Compute comprehensive calibration diagnostics.
    
    Computes pre/post calibration metrics:
        - Brier Score (per-sample squared error)
        - Expected Calibration Error (frequency calibration)
        - AD-weighted diagnostics
    
    Args:
        y_prob_raw: Raw model predictions
        y_prob_calibrated: Calibrated predictions (optional)
        y_true: True labels
        ad_score: Applicability domain scores
        eps: Epsilon for numerical stability
    
    Returns:
        Dictionary with all diagnostic metrics
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob_raw = np.clip(np.asarray(y_prob_raw, dtype=np.float64), eps, 1.0 - eps)
    ad_score = np.clip(np.asarray(ad_score, dtype=np.float64), 0.0, 1.0)
    
    diagnostics: Dict[str, Any] = {
        "n_samples": int(len(y_true)),
        "raw": {
            "brier_score": _compute_brier_score(y_true, y_prob_raw),
            "ece": _compute_expected_calibration_error(y_true, y_prob_raw),
        },
    }
    
    # Add calibrated diagnostics if available
    if y_prob_calibrated is not None:
        y_prob_calibrated = np.clip(np.asarray(y_prob_calibrated, dtype=np.float64), eps, 1.0 - eps)
        diagnostics["calibrated"] = {
            "brier_score": _compute_brier_score(y_true, y_prob_calibrated),
            "ece": _compute_expected_calibration_error(y_true, y_prob_calibrated),
        }
        # Compute improvement
        diagnostics["improvement"] = {
            "brier_score_delta": diagnostics["raw"]["brier_score"] - diagnostics["calibrated"]["brier_score"],
            "ece_delta": diagnostics["raw"]["ece"] - diagnostics["calibrated"]["ece"],
        }
    
    # Stratify diagnostics by AD_Score confidence
    in_domain_mask = ad_score >= 0.5
    if np.any(in_domain_mask):
        diagnostics["in_domain"] = {
            "n": int(np.sum(in_domain_mask)),
            "rate": float(np.mean(in_domain_mask)),
            "brier_score": _compute_brier_score(y_true[in_domain_mask], y_prob_raw[in_domain_mask]),
            "ece": _compute_expected_calibration_error(y_true[in_domain_mask], y_prob_raw[in_domain_mask]),
        }
    
    out_domain_mask = ad_score < 0.5
    if np.any(out_domain_mask):
        diagnostics["out_domain"] = {
            "n": int(np.sum(out_domain_mask)),
            "rate": float(np.mean(out_domain_mask)),
            "brier_score": _compute_brier_score(y_true[out_domain_mask], y_prob_raw[out_domain_mask]),
            "ece": _compute_expected_calibration_error(y_true[out_domain_mask], y_prob_raw[out_domain_mask]),
        }
    
    return diagnostics


# %%
# 🆕 Helper function for calibration detection and config update
def _apply_detected_calibration_config(
    config: ADConfig,
    run_dir: Path,
    split_seed: int,
) -> ADConfig:
    """根据 step20 的输出自动调整配置"""
    if not _CALIBRATION_INTEGRATION_AVAILABLE or try_load_calibrated_predictions is None:
        return config
    
    calib_result = try_load_calibrated_predictions(
        run_dir=run_dir,
        split_seed=split_seed,
        model_key=config.model_key,
        calibration_method=config.calibration_method
    )
    
    if calib_result is not None:
        _, metadata = calib_result
        config.calibration_available = True
        config.calibration_metadata = metadata
        
        # Adjust shrinkage strategy based on calibration availability
        if config.logit_shrinkage_method == "auto":
            config.logit_shrinkage_method = "probability_space"  # Preserve calibration
            print(
                f"✓ Calibration detected ({metadata.get('calibration_method_used')}). "
                f"Using probability_space shrinkage to preserve calibration."
            )
        else:
            print(
                f"⚠️  Calibration available but logit_shrinkage_method={config.logit_shrinkage_method}. "
                f"This may affect calibration properties."
            )
    else:
        if config.logit_shrinkage_method == "auto":
            config.logit_shrinkage_method = "conservative"
            print(
                "No calibration detected from step20. "
                "Using conservative shrinkage method."
            )
    
    return config


# %%
def compute_and_export(config: ADConfig) -> Dict[str, Any]:
    run_dir = config.run_dir
    split_seed = int(config.split_seed)
    split_dir = Path(config.run_dir) / f"split_seed_{config.split_seed}"

    ad_output_dir = (
        split_dir / 
        "validation" / 
        "applicability_domain" / 
        config.model_key /   
        f"seed_{config.split_seed}"
    )
    
    ad_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"AD artifacts will be saved to: {ad_output_dir}")

    # Step 0: Auto-detect calibration configuration
    if config.detect_calibration and _CALIBRATION_INTEGRATION_AVAILABLE:
        config = _apply_detected_calibration_config(config, run_dir, split_seed)

    train_npz, external_npz, predictions_csv = _default_paths(run_dir, split_seed)
    train_npz = config.train_npz or train_npz
    external_npz = config.external_npz or external_npz
    predictions_csv = config.predictions_csv or predictions_csv

    output_dir = config.output_dir or (split_dir / "validation" / "applicability_domain" / config.model_key / f"seed_{split_seed}")
    output_dir.mkdir(parents=True, exist_ok=True)


    # Load split arrays (fingerprints + descriptors are stored separately).
    train = _load_npz(train_npz)
    ext = _load_npz(external_npz)
    fp_train = np.asarray(train["fp"], dtype=np.float32)
    desc_train = np.asarray(train["desc"], dtype=np.float32)
    fp_ext = np.asarray(ext["fp"], dtype=np.float32)
    desc_ext = np.asarray(ext["desc"], dtype=np.float32)

    # Assert shape consistency
    assert fp_train.shape[1] == fp_ext.shape[1], "Fingerprint dimension mismatch"
    assert desc_train.shape[1] == desc_ext.shape[1], "Descriptor dimension mismatch"
    assert fp_train.shape[0] == desc_train.shape[0], "Train FP/desc row mismatch"
    assert fp_ext.shape[0] == desc_ext.shape[0], "External FP/desc row mismatch"

    # ===== NEW: Load and apply fingerprint mask for feature alignment =====
    fp_mask = None
    fp_mask_path = split_dir / "feature_processors" / "fp_mask.npy"
    
    if fp_mask_path.exists():
        try:
            fp_mask = np.load(fp_mask_path)
            assert fp_mask.dtype == bool or fp_mask.dtype == np.bool_, f"Expected boolean mask, got {fp_mask.dtype}"
            assert len(fp_mask) == fp_train.shape[1], f"Mask size {len(fp_mask)} != FP dimension {fp_train.shape[1]}"
            
            # Apply mask to fingerprints: select only the filtered features
            fp_train_orig_dim = fp_train.shape[1]
            fp_train = fp_train[:, fp_mask].astype(np.float32)
            fp_ext = fp_ext[:, fp_mask].astype(np.float32)
            fp_aligned_dim = fp_train.shape[1]
            
            logging.info(
                f"✓ Fingerprint alignment applied: "
                f"{fp_train_orig_dim} → {fp_aligned_dim} bits using fp_mask.npy"
            )
            print(
                f"Aligned fingerprints from {fp_train_orig_dim} to {fp_aligned_dim} bits using fp_mask.npy"
            )
        except Exception as e:
            logging.warning(f"Failed to load/apply fp_mask.npy: {e}. Proceeding with full fingerprints.")
            print(f"⚠️ Warning: Could not apply fingerprint mask: {e}")
            fp_mask = None
    else:
        logging.info(
            f"No fp_mask.npy found at {fp_mask_path}. "
            f"Using full {fp_train.shape[1]}-bit fingerprints (backward compatibility mode)."
        )

    # Descriptor space is used for cosine similarity (to avoid redundancy with fingerprint-based Tanimoto).
    desc_scaler = StandardScaler()
    desc_train_scaled = desc_scaler.fit_transform(desc_train)
    desc_ext_scaled = desc_scaler.transform(desc_ext)

    # Base-domain feature space (for leverage / mahalanobis / kNN density / SOM).
    base_space = str(config.base_feature_space).strip().lower()
    if base_space == "desc":
        X_train_base = desc_train
        X_ext_base = desc_ext
    elif base_space == "fingerprint":
        X_train_base = fp_train
        X_ext_base = fp_ext
    elif base_space == "full":
        X_train_base = np.concatenate([fp_train, desc_train], axis=1)
        X_ext_base = np.concatenate([fp_ext, desc_ext], axis=1)
    else:
        raise ValueError(f"Unknown base_feature_space: {config.base_feature_space!r} (expected: desc/fingerprint/full)")

    assert X_train_base.shape[1] == X_ext_base.shape[1], "Base feature dimension mismatch"

    base_scaler = StandardScaler()
    X_train_base_scaled = base_scaler.fit_transform(X_train_base)
    X_ext_base_scaled = base_scaler.transform(X_ext_base)

    # Align predictions (external only) for residual computation and export.
    aligned = _align_predictions_to_external(
        predictions_csv=predictions_csv,
        external_ids=np.asarray(ext["id"]),
        external_smiles=np.asarray(ext["smiles"]),
        model_key=config.model_key,
        id_column=config.id_column,
        smiles_column=config.smiles_column,
        model_column=config.model_column,
    )

    # 🆕 Step D: Flexible column resolution with auto-calibration detection
    # TASK 5: Feature alignment - check feature_names_final.json for consistency
    feature_names_path = split_dir / "data" / "feature_names_final.json"
    feature_names_metadata: Optional[Dict[str, Any]] = None
    if feature_names_path.exists():
        try:
            with open(feature_names_path) as f:
                feature_names_metadata = json.load(f)
            logging.info(f"✓ Loaded feature metadata: {len(feature_names_metadata)} features")
        except Exception as e:
            logging.warning(f"Failed to load feature_names_final.json: {e}")
    
    # Try to find calibrated probability column first
    y_prob_source = "original"
    
    if _CALIBRATION_INTEGRATION_AVAILABLE and flexible_column_resolution is not None:
        # Try multi-level priority search
        _, y_true = flexible_column_resolution(
            aligned,
            priority_names=[config.y_true_column, "y_true"],
            friendly_name="true labels"
        )
        
        # Search for probability with priority on calibrated versions
        y_prob_col, y_prob = flexible_column_resolution(
            aligned,
            priority_names=[
                "y_prob_calibrated",     # Step20 calibrated
                "calibrated_prob",
                "y_prob_cal",
                config.y_prob_column,    # Configuration
                "y_prob",                # Fallback
            ],
            friendly_name="predicted probabilities",
            required=True
        )
        
        if "calibrated" in y_prob_col.lower():
            y_prob_source = "calibrated"
    else:
        # Fallback to original hard-coded logic
        for col in (config.y_true_column, config.y_prob_column):
            if col not in aligned.columns:
                raise ValueError(f"Predictions missing '{col}' column: {predictions_csv}")

        y_true = aligned[config.y_true_column].astype(float).to_numpy()
        y_prob = aligned[config.y_prob_column].astype(float).to_numpy()

    # Classification diagnostics: log-loss / deviance residual (more meaningful than y_true - y_prob).
    log_loss = _per_sample_log_loss(y_true=y_true, y_prob=y_prob)
    dev_resid = _deviance_residual(y_true=y_true, y_prob=y_prob)
    std_resid = _zscore(dev_resid)
    prob_error = (y_prob.astype(np.float64) - y_true.astype(np.float64))
    abs_prob_error = np.abs(prob_error)
    williams_outlier = np.abs(std_resid) > float(config.williams_residual_z)

    # Base-domain flags.
    base_method = str(config.base_method).strip().lower()
    leverage = np.full(len(X_ext_base_scaled), np.nan, dtype=np.float64)
    h_star = float("nan")
    pca_components = 0
    mahalanobis_dist = np.full(len(X_ext_base_scaled), np.nan, dtype=np.float64)
    mahalanobis_thr = float("nan")
    knn_mean_dist = np.full(len(X_ext_base_scaled), np.nan, dtype=np.float64)
    knn_thr = float("nan")
    knn_train_dist = None  # For computing density score
    density_reference_raw = None

    if base_method == "leverage":
        leverage, h_star, pca_components = _compute_leverage_pca(
            X_train_scaled=X_train_base_scaled,
            X_query_scaled=X_ext_base_scaled,
            variance_ratio=config.leverage_pca_variance,
            fixed_components=config.leverage_pca_components,
            ad_output_dir=ad_output_dir,
            scaler_to_save=base_scaler
        )
        leverage_train, _, _ = _compute_leverage_pca(
            X_train_scaled=X_train_base_scaled,
            X_query_scaled=X_train_base_scaled,
            variance_ratio=config.leverage_pca_variance,
            fixed_components=config.leverage_pca_components,
        )
        density_reference_raw = leverage_train
        base_in_domain = leverage <= h_star
    elif base_method == "mahalanobis":
        d_train, d_ext = _mahalanobis_distance(X_train_base_scaled, X_ext_base_scaled)
        mahalanobis_dist = d_ext
        density_reference_raw = d_train
        mahalanobis_thr = _quantile_threshold(d_train, config.domain_threshold_quantile)
        base_in_domain = mahalanobis_dist <= mahalanobis_thr
    elif base_method == "knn_density":
        d_train, d_ext = _knn_mean_distance(X_train_base_scaled, X_ext_base_scaled, k=config.knn_k)
        knn_mean_dist = d_ext
        knn_train_dist = d_train  # Store for density score computation
        density_reference_raw = d_train
        knn_thr = _quantile_threshold(d_train, config.domain_threshold_quantile)
        base_in_domain = knn_mean_dist <= knn_thr
    else:
        raise ValueError(f"Unknown base_method: {config.base_method!r} (expected: leverage/mahalanobis/knn_density)")

    # SOM occupancy flags (optional).
    som_in_domain = np.ones(len(X_ext_base_scaled), dtype=bool)
    som = None
    if not config.skip_som:
        som, occupied = _train_som(X_train_base_scaled, config.som_rows, config.som_cols, config.som_iterations)
        som_in_domain = _som_flags(som, occupied, X_ext_base_scaled)

    # Similarity-based flags.
    tanimoto_max = _tanimoto_max(_binarize_fp(fp_train), _binarize_fp(fp_ext))
    tanimoto_in_domain = tanimoto_max >= float(config.tanimoto_threshold)

    # Cosine similarity is computed on DESCRIPTORS ONLY (fingerprint signal is already covered by Tanimoto).
    if len(desc_train_scaled) * len(desc_ext_scaled) < 2_000_000:
        cosine_mat = cosine_similarity(desc_ext_scaled, desc_train_scaled)
        cosine_max = np.max(cosine_mat, axis=1).astype(np.float32)
    else:
        cosine_max = _cosine_max_blockwise(desc_ext_scaled, desc_train_scaled, block_size=config.cosine_block_size)
    cosine_in_domain = cosine_max >= float(config.cosine_threshold)

    base_domain = base_in_domain & som_in_domain
    similarity_domain = tanimoto_in_domain & cosine_in_domain
    if config.strict_similarity:
        in_domain = base_domain & similarity_domain
    else:
        in_domain = base_domain | similarity_domain

    # ===== TASK 1: Learn optimal weights from dev set =====
    # Initialize with defaults or manual config
    w1 = config.w1_tanimoto if config.w1_tanimoto is not None else 0.7  # Tanimoto weight
    w2 = config.w2_cosine if config.w2_cosine is not None else 0.3    # Cosine weight
    w3 = config.w3_similarity if config.w3_similarity is not None else 0.6  # Similarity weight
    w4 = config.w4_density if config.w4_density is not None else 0.4    # Density weight
    
    weight_config = None
    cal_curve_train = None
    
    # Log manual weight config if any is specified
    manual_weights = [config.w1_tanimoto, config.w2_cosine, config.w3_similarity, config.w4_density]
    if any(w is not None for w in manual_weights):
        logging.info(
            f"Manual weight configuration: w1={w1:.3f}, w2={w2:.3f}, w3={w3:.3f}, w4={w4:.3f}"
        )
        print(f"Using manual weights: w1={w1:.3f}, w2={w2:.3f}, w3={w3:.3f}, w4={w4:.3f}")

    if config.learn_weights:
        # Load dev set predictions for weight optimization
        dev_predictions_csv = split_dir / "predictions" / "dev_train_predictions.csv"
        if dev_predictions_csv.exists():
            dev_preds = pd.read_csv(dev_predictions_csv)
            dev_mask = dev_preds[config.model_column].astype(str).str.upper() == str(config.model_key).upper()
            dev_preds = dev_preds[dev_mask].copy()

            if len(dev_preds) > 10:
                # Compute dev set metrics (avoid self-similarity for cosine)
                dev_fp = np.asarray(train["fp"], dtype=np.float32)
                dev_desc = np.asarray(train["desc"], dtype=np.float32)
                
                # Apply fingerprint mask to dev set (same as training/external)
                if fp_mask is not None:
                    dev_fp = dev_fp[:, fp_mask].astype(np.float32)
                
                dev_fp_bin = _binarize_fp(dev_fp)
                dev_desc_scaled = desc_scaler.transform(dev_desc).astype(np.float32)
                dev_tanimoto = np.zeros(len(dev_fp_bin), dtype=np.float64)
                dev_cosine = np.zeros(len(dev_desc_scaled), dtype=np.float64)
                for i in range(len(dev_fp_bin)):
                    # Leave-one-out max similarity
                    mask = np.arange(len(dev_fp_bin)) != i
                    dev_tanimoto[i] = np.max(_tanimoto_max(dev_fp_bin[mask], dev_fp_bin[i:i+1]))
                    dev_cosine[i] = np.max(cosine_similarity(dev_desc_scaled[i:i+1], dev_desc_scaled[mask]))

                # Compute dev set density (TASK 2: unified)
                if base_method == "knn_density" and knn_train_dist is not None:
                    dev_density_raw = knn_train_dist
                elif base_method == "mahalanobis":
                    d_train_maha, _ = _mahalanobis_distance(X_train_base_scaled, X_train_base_scaled)
                    dev_density_raw = d_train_maha
                elif base_method == "leverage":
                    leverage_train = density_reference_raw
                    if leverage_train is None:
                        leverage_train, _, _ = _compute_leverage_pca(
                            X_train_base_scaled, X_train_base_scaled, config.leverage_pca_variance,
                            fixed_components=config.leverage_pca_components
                        )
                    dev_density_raw = leverage_train
                else:
                    dev_density_raw = np.ones(len(X_train_base))

                dev_density = _compute_density_score_unified(
                    dev_density_raw,
                    base_method,
                    reference_arr=density_reference_raw,
                )

                # Compute error metric
                dev_y_true = dev_preds[config.y_true_column].astype(float).to_numpy()
                dev_y_prob = dev_preds[config.y_prob_column].astype(float).to_numpy()
                dev_error = _per_sample_log_loss(dev_y_true, dev_y_prob)

                # Optimize weights
                weight_config = _optimize_ad_weights(
                    dev_tanimoto[:len(dev_preds)],
                    dev_cosine[:len(dev_preds)],
                    dev_density,
                    dev_error,
                    grid_resolution=config.weight_search_grid
                )

                w1 = weight_config["w1_tanimoto"]
                w2 = weight_config["w2_cosine"]
                w3 = weight_config["w3_similarity"]
                w4 = weight_config["w4_density"]
    
    # Override with provided config if specified
    if config.ad_weight_config is not None:
        w1 = config.ad_weight_config.get("w1_tanimoto", w1)
        w2 = config.ad_weight_config.get("w2_cosine", w2)
        w3 = config.ad_weight_config.get("w3_similarity", w3)
        w4 = config.ad_weight_config.get("w4_density", w4)
    
    # ===== TASK 2: Unify density across all base methods =====
    if base_method == "knn_density" and knn_train_dist is not None:
        ext_density_raw = knn_mean_dist
    elif base_method == "mahalanobis":
        ext_density_raw = mahalanobis_dist
    elif base_method == "leverage":
        ext_density_raw = leverage
    else:
        ext_density_raw = np.ones(len(X_ext_base_scaled))
    
    density_score = _compute_density_score_unified(
        ext_density_raw,
        base_method,
        reference_arr=density_reference_raw,
    )

    # Compute continuous AD score with learned weights
    tanimoto_clipped = np.clip(tanimoto_max.astype(np.float64), 0.0, 1.0)
    cosine_clipped = np.clip(cosine_max.astype(np.float64), 0.0, 1.0)
    sim_score = w1 * tanimoto_clipped + w2 * cosine_clipped
    sim_score = np.clip(sim_score, 0.0, 1.0)

    AD_score = w3 * sim_score + w4 * density_score
    AD_score = np.clip(AD_score, 0.0, 1.0)

    # ===== TASK 3: 🆕 Apply AD shrinkage with strategy selection =====
    # TASK 5: Enhanced epsilon guards in probability/logit transformations
    
    # Determine actual shrinkage method to use
    shrinkage_method = config.logit_shrinkage_method
    if shrinkage_method == "auto":
        if config.calibration_available:
            shrinkage_method = "probability_space"
            logging.info("Auto-detected calibration; using probability_space shrinkage")
        else:
            shrinkage_method = "conservative"
            logging.info("No calibration detected; using conservative shrinkage")
    
    if shrinkage_method == "logit_space":
        logging.warning(
            "⚠️ Using logit_space shrinkage. "
            "This may affect calibration properties if using calibrated probabilities."
        )

    # Apply shrinkage based on method with epsilon guards
    eps_shrink = 1e-7
    
    # Apply AD score power (non-linear penalty)
    k = float(config.ad_score_power)
    AD_score_powered = np.power(np.clip(AD_score, eps_shrink, 1.0), k)
    logging.info(f"AD score power exponent: k={k:.2f}")
    
    try:
        if _CALIBRATION_INTEGRATION_AVAILABLE and apply_ad_shrinkage is not None:
            final_score_shrunk = apply_ad_shrinkage(
                y_prob=y_prob,
                ad_score=AD_score_powered,
                method=shrinkage_method
            )
        else:
            raise RuntimeError("Calibration integration unavailable")
    except Exception as e:
        logging.warning(f"Shrinkage method {shrinkage_method} failed: {e}. Using fallback.")
        # Fallback to probability-space scaling (safest option) with power exponent
        final_score_shrunk = np.clip(y_prob, eps_shrink, 1.0 - eps_shrink) * AD_score_powered
    
    # Keep old-style final score for backward compatibility (with power exponent)
    final_score = np.clip(y_prob.astype(np.float64), eps_shrink, 1.0 - eps_shrink) * AD_score_powered
    
    # ===== TASK 4: Compute comprehensive calibration diagnostics =====
    # Retrieve raw probabilities for comparison if calibrated version was used
    y_prob_raw = None
    if y_prob_source == "calibrated":
        try:
            preds_df_all = pd.read_csv(predictions_csv)
            model_mask_all = preds_df_all[config.model_column].astype(str).str.upper() == str(config.model_key).upper()
            y_prob_raw_col, y_prob_raw = flexible_column_resolution(
                preds_df_all[model_mask_all],
                priority_names=[config.y_prob_column, "y_prob"],
                friendly_name="raw predicted probabilities",
                required=False
            )
            if y_prob_raw is not None:
                y_prob_raw = y_prob_raw[:len(y_true)]  # Align lengths
        except Exception as e:
            logging.warning(f"Could not retrieve raw probabilities: {e}")
    
    # Compute diagnostics
    calibration_diagnostics = _compute_calibration_diagnostics(
        y_prob_raw=y_prob_raw if y_prob_raw is not None else y_prob,
        y_prob_calibrated=y_prob if y_prob_source == "calibrated" else None,
        y_true=y_true,
        ad_score=AD_score,
        eps=eps_shrink
    )
    
    logging.info(
        f"✓ Calibration diagnostics computed:\n"
        f"  - Raw Brier: {calibration_diagnostics['raw']['brier_score']:.4f}\n"
        f"  - Raw ECE: {calibration_diagnostics['raw']['ece']:.4f}"
    )
    
    if "calibrated" in calibration_diagnostics:
        logging.info(
            f"  - Calibrated Brier: {calibration_diagnostics['calibrated']['brier_score']:.4f}\n"
            f"  - Calibrated ECE: {calibration_diagnostics['calibrated']['ece']:.4f}\n"
            f"  - Brier improvement: {calibration_diagnostics['improvement']['brier_score_delta']:.4f}"
        )
    
    # Correlation diagnostics
    ad_error_corr = float(np.corrcoef(AD_score, log_loss)[0, 1]) if len(AD_score) > 1 else 0.0
    ad_error_corr = ad_error_corr if not np.isnan(ad_error_corr) else 0.0
    
    # Compute calibration curve with proper diagnostics
    cal_curve_ext = _compute_calibration_curve(AD_score, log_loss, n_bins=10, eps=eps_shrink)
    out_df = aligned.copy()
    out_df["Base_Method"] = base_method
    out_df["Base_Feature_Space"] = base_space
    out_df["Leverage"] = leverage
    out_df["Leverage_h_star"] = h_star
    out_df["Mahalanobis_Dist"] = mahalanobis_dist
    out_df["Mahalanobis_Threshold"] = mahalanobis_thr
    out_df["KNN_MeanDist"] = knn_mean_dist
    out_df["KNN_Threshold"] = knn_thr
    out_df["Base_In_Domain"] = base_in_domain

    out_df["LogLoss"] = log_loss
    out_df["DevianceResidual"] = dev_resid
    out_df["StdResidual"] = std_resid
    out_df["ProbError"] = prob_error
    out_df["AbsProbError"] = abs_prob_error
    out_df["Williams_Outlier"] = williams_outlier
    out_df["SOM_In_Domain"] = som_in_domain
    out_df["Tanimoto_max"] = tanimoto_max
    out_df["Tanimoto_In_Domain"] = tanimoto_in_domain
    out_df["Cosine_max"] = cosine_max
    out_df["Cosine_In_Domain"] = cosine_in_domain
    out_df["In_Domain"] = in_domain

    # Add continuous scoring columns
    out_df["Sim_Score"] = sim_score
    out_df["Density_Score"] = density_score
    out_df["AD_Score"] = AD_score
    out_df["Final_Score"] = final_score

    # Always add AD_Score_Learned, Final_Score_Shrunk, Weight_Config_ID
    out_df["AD_Score_Learned"] = AD_score
    out_df["Final_Score_Shrunk"] = final_score_shrunk
    # Keep backward-compatible name
    out_df["Final_Score_Logit"] = final_score_shrunk
    
    weight_config_id = None
    if weight_config and "config_id" in weight_config:
        weight_config_id = weight_config["config_id"]
    else:
        weight_config_id = f"{base_method}_{config.model_key}_seed{split_seed}"
    out_df["Weight_Config_ID"] = weight_config_id

    ad_table_path = output_dir / "ad_external_predictions.csv"
    out_df.to_csv(ad_table_path, index=False)

    # TASK 4: Export calibration curve
    cal_curve_path = output_dir / "ad_calibration_curve.csv"
    cal_curve_ext.to_csv(cal_curve_path, index=False)

    # Optionally update the original predictions file (with backup).
    if config.inplace_update_predictions:
        preds_all = pd.read_csv(predictions_csv)
        model_mask = preds_all[config.model_column].astype(str).str.upper() == str(config.model_key).upper()
        if int(model_mask.sum()) != len(out_df):
            raise ValueError("Inplace update requires 1:1 model rows with external samples; counts do not match.")
        _backup_file(predictions_csv)
        preds_all.loc[model_mask, out_df.columns] = out_df.to_numpy()
        preds_all.to_csv(predictions_csv, index=False)

    summary = {
        "run_dir": str(run_dir),
        "split_seed": split_seed,
        "model_key": config.model_key,
        "train_npz": str(train_npz),
        "external_npz": str(external_npz),
        "predictions_csv": str(predictions_csv),
        "output_dir": str(output_dir),
        "n_train": int(len(X_train_base)),
        "n_external": int(len(X_ext_base)),
        "base_feature_dim": int(X_train_base.shape[1]),
        "fp_dim": int(fp_train.shape[1]),
        "desc_dim": int(desc_train.shape[1]),
        "base_method": base_method,
        "base_feature_space": base_space,
        "domain_threshold_quantile": float(config.domain_threshold_quantile),
        "knn_k": int(config.knn_k),
        "leverage_pca_variance": float(config.leverage_pca_variance),
        "h_star": float(h_star),
        "pca_components": int(pca_components),
        "williams_residual_z": float(config.williams_residual_z),
        "tanimoto_threshold": float(config.tanimoto_threshold),
        "cosine_threshold": float(config.cosine_threshold),
        "strict_similarity": bool(config.strict_similarity),
        "skip_som": bool(config.skip_som),
        "som_rows": int(config.som_rows),
        "som_cols": int(config.som_cols),
        "som_iterations": int(config.som_iterations),
        "rates": {
            "base_in_domain": float(np.mean(base_in_domain)) if len(base_in_domain) else 0.0,
            "som_in_domain": float(np.mean(som_in_domain)) if len(som_in_domain) else 0.0,
            "tanimoto_in_domain": float(np.mean(tanimoto_in_domain)) if len(tanimoto_in_domain) else 0.0,
            "cosine_in_domain": float(np.mean(cosine_in_domain)) if len(cosine_in_domain) else 0.0,
            "in_domain": float(np.mean(in_domain)) if len(in_domain) else 0.0,
        },
        "continuous_ad_scores": {
            "ad_score_power": float(config.ad_score_power),
            "density_reference_median": float(np.median(np.asarray(density_reference_raw, dtype=np.float64)))
            if density_reference_raw is not None and len(np.asarray(density_reference_raw).reshape(-1)) > 0
            else None,
            "sim_score_weights": {"tanimoto": float(w1), "cosine": float(w2)},
            "ad_score_weights": {"similarity": float(w3), "density": float(w4)},
            "weight_learning_config": weight_config if weight_config else {"w1": w1, "w2": w2, "w3": w3, "w4": w4},
            "weights_learned": bool(config.learn_weights and weight_config is not None),
            "weight_optimization_grid": int(config.weight_search_grid) if config.learn_weights else None,
            "sim_score_mean": float(np.mean(sim_score)) if len(sim_score) else 0.0,
            "sim_score_std": float(np.std(sim_score)) if len(sim_score) else 0.0,
            "sim_score_min": float(np.min(sim_score)) if len(sim_score) else 0.0,
            "sim_score_max": float(np.max(sim_score)) if len(sim_score) else 0.0,
            "density_score_mean": float(np.mean(density_score)) if len(density_score) else 0.0,
            "density_score_std": float(np.std(density_score)) if len(density_score) else 0.0,
            "density_score_min": float(np.min(density_score)) if len(density_score) else 0.0,
            "density_score_max": float(np.max(density_score)) if len(density_score) else 0.0,
            "ad_score_mean": float(np.mean(AD_score)) if len(AD_score) else 0.0,
            "ad_score_std": float(np.std(AD_score)) if len(AD_score) else 0.0,
            "ad_score_min": float(np.min(AD_score)) if len(AD_score) else 0.0,
            "ad_score_max": float(np.max(AD_score)) if len(AD_score) else 0.0,
            "final_score_mean": float(np.mean(final_score)) if len(final_score) else 0.0,
            "final_score_std": float(np.std(final_score)) if len(final_score) else 0.0,
            "final_score_min": float(np.min(final_score)) if len(final_score) else 0.0,
            "final_score_max": float(np.max(final_score)) if len(final_score) else 0.0,
            "final_score_shrunk_mean": float(np.mean(final_score_shrunk)) if len(final_score_shrunk) else 0.0,
            "ad_error_correlation": float(ad_error_corr),
        },
        # TASK 4: Enhanced calibration diagnostics
        "calibration_diagnostics": calibration_diagnostics,
        # 🆕 Enhanced calibration metadata
        "calibration": {
            "detected": bool(config.calibration_available),
            "metadata": config.calibration_metadata if config.calibration_available else None,
            "shrinkage_method": config.logit_shrinkage_method,
            "shrinkage_method_used": shrinkage_method,
            "n_bins": 10,
            "calibration_curve_path": str(cal_curve_path),
            "binned_statistics": cal_curve_ext.to_dict("records") if len(cal_curve_ext) > 0 else [],
        },
        # TASK 5: Feature engineering tracking
        "feature_engineering": {
            "feature_metadata_file": str(feature_names_path) if feature_names_path.exists() else None,
            "feature_metadata_loaded": feature_names_metadata is not None,
            "epsilon_guards_enabled": True,
            "epsilon_value": float(eps_shrink),
        },
        # 🆕 New: explicit probability source tracking
        "probability_source": y_prob_source,
        "exports": {
            "ad_external_predictions_csv": str(ad_table_path),
            "ad_calibration_curve_csv": str(cal_curve_path),
        },
    }

    (output_dir / "ad_summary.json").write_text(json.dumps(summary, indent=2))
    
    # TASK 1: Save learned weights config (always write, even if defaults)
    weight_config_path = output_dir / "ad_weight_config.json"
    if not weight_config:
        # Write default weights if not learned
        weight_config = {
            "w1_tanimoto": float(w1),
            "w2_cosine": float(w2),
            "w3_similarity": float(w3),
            "w4_density": float(w4),
            "correlation": float(ad_error_corr),
        }
    # Guarantee all required fields
    for k in ["w1_tanimoto", "w2_cosine", "w3_similarity", "w4_density", "correlation"]:
        if k not in weight_config:
            weight_config[k] = 0.0
    weight_config["config_id"] = f"{base_method}_{config.model_key}_seed{split_seed}"
    weight_config["timestamp"] = str(pd.Timestamp.now())
    weight_config_path.write_text(json.dumps(weight_config, indent=2))

    # Store a compact npz so plotting cells can load without redoing heavy work.
    np.savez_compressed(
        output_dir / "ad_plot_data.npz",
        leverage=leverage,
        std_resid=std_resid,
        in_domain=in_domain.astype(np.int8),
        sim_score=sim_score.astype(np.float32),
        density_score=density_score.astype(np.float32),
        ad_score=AD_score.astype(np.float32),
        final_score=final_score.astype(np.float32),
        final_score_shrunk=final_score_shrunk.astype(np.float32),
        log_loss=log_loss.astype(np.float32),
        X_train_base_scaled=X_train_base_scaled.astype(np.float32),
        X_ext_base_scaled=X_ext_base_scaled.astype(np.float32),
        desc_train_scaled=desc_train_scaled.astype(np.float32),
        desc_ext_scaled=desc_ext_scaled.astype(np.float32),
    )

    # Plots are optional (can be heavy for t-SNE).
    if config.make_plots:
        from matplotlib import pyplot as plt
        from matplotlib.ticker import MaxNLocator
        from sklearn.manifold import TSNE

        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.serif": ["Cambria", "Times New Roman", "Times", "DejaVu Serif"],
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.grid": True,
                "grid.linestyle": ":",
                "grid.alpha": 0.35,
            }
        )

        def _style_axis(ax) -> None:
            ax.grid(True, linestyle=":", alpha=0.35)
            ax.tick_params(direction="out", length=4, width=1)

        def _plot_williams(x_arr: np.ndarray,
                           resid_arr: np.ndarray,
                           x_threshold: float,
                           x_label: str,
                           out_base: Path) -> None:
            fig, ax = plt.subplots(figsize=(4.2, 3.2), constrained_layout=True)
            is_lev = x_arr > x_threshold
            is_res = np.abs(resid_arr) > float(config.williams_residual_z)
            in_d = (~is_lev) & (~is_res)

            ax.scatter(x_arr[in_d], resid_arr[in_d], s=18, alpha=0.55, label="In-domain")
            ax.scatter(x_arr[is_lev & ~is_res], resid_arr[is_lev & ~is_res], s=28, marker="D", label="High distance")
            ax.scatter(x_arr[~is_lev & is_res], resid_arr[~is_lev & is_res], s=26, marker="o", label="High residual")
            ax.scatter(x_arr[is_lev & is_res], resid_arr[is_lev & is_res], s=40, marker="x", label="Critical")

            ax.axvline(x_threshold, linestyle="--", linewidth=1.0, color="black", label=f"thr={x_threshold:.2f}")
            ax.axhline(float(config.williams_residual_z), linestyle=":", linewidth=1.0, color="crimson", alpha=0.8)
            ax.axhline(-float(config.williams_residual_z), linestyle=":", linewidth=1.0, color="crimson", alpha=0.8)
            ax.set_xlabel(x_label)
            ax.set_ylabel("Std. deviance residual (z)")
            ax.set_title("AD Diagnostic Plot")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4, prune="both"))
            _style_axis(ax)
            ax.legend(loc="best", fontsize=8, frameon=False)
            fig.savefig(out_base.with_suffix(".png"), dpi=300)
            fig.savefig(out_base.with_suffix(".svg"))
            plt.close(fig)

        def _plot_pca_tsne(Xtr: np.ndarray, Xte: np.ndarray, in_d: np.ndarray, out_base: Path) -> None:
            pca2 = PCA(n_components=2)
            tr_p = pca2.fit_transform(Xtr)
            te_p = pca2.transform(Xte)

            tsne = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=42)
            emb = tsne.fit_transform(np.vstack([Xtr, Xte]))
            tr_t = emb[: len(Xtr)]
            te_t = emb[len(Xtr) :]

            fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
            axes[0].scatter(tr_p[:, 0], tr_p[:, 1], s=10, alpha=0.25, label="Train")
            axes[0].scatter(te_p[in_d, 0], te_p[in_d, 1], s=24, marker="D", label="External (in)")
            axes[0].scatter(te_p[~in_d, 0], te_p[~in_d, 1], s=24, marker="D", label="External (out)")
            axes[0].set_title("PCA (scaled features)")
            axes[0].set_xlabel("PC1")
            axes[0].set_ylabel("PC2")
            _style_axis(axes[0])

            axes[1].scatter(tr_t[:, 0], tr_t[:, 1], s=10, alpha=0.25, label="Train")
            axes[1].scatter(te_t[in_d, 0], te_t[in_d, 1], s=24, marker="D", label="External (in)")
            axes[1].scatter(te_t[~in_d, 0], te_t[~in_d, 1], s=24, marker="D", label="External (out)")
            axes[1].set_title("t-SNE (scaled features)")
            axes[1].set_xlabel("Dim 1")
            axes[1].set_ylabel("Dim 2")
            _style_axis(axes[1])

            axes[1].legend(loc="best", fontsize=8, frameon=False)
            fig.savefig(out_base.with_suffix(".png"), dpi=300)
            fig.savefig(out_base.with_suffix(".svg"))
            plt.close(fig)

        if base_method == "leverage":
            x = leverage
            thr = h_star
            xlabel = "Leverage (PCA)"
        elif base_method == "mahalanobis":
            x = mahalanobis_dist
            thr = mahalanobis_thr
            xlabel = "Mahalanobis distance"
        else:
            x = knn_mean_dist
            thr = knn_thr
            xlabel = f"kNN mean distance (k={int(config.knn_k)})"

        _plot_williams(x, std_resid, thr, xlabel, output_dir / "ad_diagnostic")
        _plot_pca_tsne(X_train_base_scaled, X_ext_base_scaled, in_domain, output_dir / "pca_tsne")

        if som is not None:
            dist_map = som.distance_map()
            fig, ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
            im = ax.imshow(dist_map, cmap="magma", origin="lower")
            winners_tr = np.array([som.winner(x) for x in X_train_base_scaled])
            winners_te = np.array([som.winner(x) for x in X_ext_base_scaled])
            ax.scatter(winners_tr[:, 1], winners_tr[:, 0], s=12, alpha=0.8, label="Train")
            ax.scatter(winners_te[:, 1], winners_te[:, 0], s=12, marker="x", alpha=0.9, label="External")
            ax.set_title("SOM U-Matrix (occupancy)")
            ax.set_xlabel("Columns")
            ax.set_ylabel("Rows")
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)
            fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02).set_label("Inter-neuron distance")
            fig.savefig((output_dir / "som_umatrix").with_suffix(".png"), dpi=300)
            fig.savefig((output_dir / "som_umatrix").with_suffix(".svg"))
            plt.close(fig)

    return summary


# %%
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Applicability Domain analysis for step10 QSAR runs")
    p.add_argument("--run-dir", type=Path, help="Run directory (models_out/qsar_ml_YYYYMMDD_HHMMSS). Defaults to latest.")
    p.add_argument("--split-seed", type=int, required=True, help="Split seed (corresponds to split_seed_<N> folder)")
    p.add_argument("--model", required=True, help="Model key (e.g. SVC, ETC, XGBC, RFC, LR, MLP)")

    p.add_argument("--train-npz", type=Path, help="Override dev_train.npz path")
    p.add_argument("--external-npz", type=Path, help="Override external_test.npz path")
    p.add_argument("--predictions", type=Path, help="Override external_test_predictions.csv path")

    p.add_argument("--tanimoto-threshold", type=float, default=0.60)
    p.add_argument("--cosine-threshold", type=float, default=0.70)
    p.add_argument(
        "--strict-similarity",
        action="store_true",
        dest="strict_similarity",
        help="If set: in_domain = (leverage & som) AND (tanimoto & cosine). Otherwise uses OR.",
    )
    p.add_argument(
        "--no-strict-similarity",
        action="store_false",
        dest="strict_similarity",
        help="Default behavior: in_domain = (leverage & som) OR (tanimoto & cosine).",
    )
    p.set_defaults(strict_similarity=False)

    p.add_argument(
        "--base-method",
        choices=["leverage", "mahalanobis", "knn_density"],
        default="knn_density",
        help="Base AD method (in a chosen feature space).",
    )
    p.add_argument(
        "--base-feature-space",
        choices=["desc", "fingerprint", "full"],
        default="full",
        help='Feature space for base AD method: "desc" uses descriptors only; "fingerprint" uses fingerprint only; "full" uses fingerprint+desc.',
    )
    p.add_argument(
        "--domain-threshold-quantile",
        type=float,
        default=0.90,   # 
        help="Train-set quantile used as threshold for Mahalanobis/kNN density base methods.",
    )
    p.add_argument("--knn-k", type=int, default=5, help="k for kNN density base method")

    p.add_argument("--leverage-pca-variance", type=float, default=0.95)
    p.add_argument(
        "--leverage-pca-components",
        type=int,
        default=None,
        help="Fixed number of PCA components for leverage calculation. If set, overrides --leverage-pca-variance"
    )
    p.add_argument("--williams-z", type=float, default=3.0)

    p.add_argument("--skip-som", action="store_true", default=True)
    p.add_argument("--use-som", action="store_false", dest="skip_som", help="Enable SOM occupancy (requires minisom)")
    p.add_argument("--som-rows", type=int, default=12)
    p.add_argument("--som-cols", type=int, default=12)
    p.add_argument("--som-iterations", type=int, default=5000)

    p.add_argument("--output-dir", type=Path, help="Output directory. Defaults under split_seed_*/validation/...")
    p.add_argument("--inplace", action="store_true", help="Update predictions CSV in-place (writes .bak backup first)")
    p.add_argument("--make-plots", action="store_true", help="Generate plots (can be slow due to t-SNE)")
    p.add_argument("--cosine-block-size", type=int, default=1024)
    
    # TASK 1: Weight optimization arguments
    p.add_argument("--learn-weights", action="store_true", dest="learn_weights",
                   help="Learn AD weights from dev set using grid search")
    p.add_argument(
        "--calibration-method",
        choices=["isotonic", "sigmoid"],
        default="isotonic",
        help="Calibration method used for model probabilities (isotonic or sigmoid)"
    )
    p.add_argument("--weight-search-grid", type=int, default=20,
                   help="Grid resolution for weight search (higher = finer but slower)")
    
    # Manual weight configuration (when learn_weights=False)
    p.add_argument(
        "--w1-tanimoto",
        type=float,
        default=None,
        help="Manual Tanimoto weight (0.0-1.0). Ignored if --learn-weights is set. Default: 0.7"
    )
    p.add_argument(
        "--w2-cosine",
        type=float,
        default=None,
        help="Manual Cosine weight (0.0-1.0). Ignored if --learn-weights is set. Default: 0.3"
    )
    p.add_argument(
        "--w3-similarity",
        type=float,
        default=None,
        help="Manual Similarity weight (0.0-1.0). Ignored if --learn-weights is set. Default: 0.6"
    )
    p.add_argument(
        "--w4-density",
        type=float,
        default=None,
        help="Manual Density weight (0.0-1.0). Ignored if --learn-weights is set. Default: 0.4"
    )
    
    # AD Score Power (non-linear penalty)
    p.add_argument(
        "--ad-score-power",
        type=float,
        default=2.0,
        help="Power exponent for AD score shrinkage: Final_Score = Prob * (AD_Score)^k. Default: 2.0. Use 1.0 for linear, 3.0 for stronger penalty."
    )
    
    # Calibration integration arguments
    p.add_argument(
        "--logit-shrinkage-method",
        choices=["auto", "probability_space", "logit_space", "conservative", "none"],
        default="auto",
        help=(
            "Apply AD shrinkage strategy. "
            "auto: auto-detect based on calibration; "
            "probability_space: p*AD (preserve calibration); "
            "logit_space: logit space (may damage calibration); "
            "conservative: mixed; "
            "none: no shrinkage"
        )
    )
    
    p.add_argument(
        "--no-detect-calibration",
        action="store_false",
        dest="detect_calibration",
        help="Disable auto-detection of step20 calibration"
    )
    p.set_defaults(detect_calibration=True)
    
    p.add_argument(
        "--compare-calibration",
        action="store_true",
        dest="compare_pre_post_calibration",
        help="Generate calibration pre/post comparison curves (additional computation)"
    )
    p.set_defaults(compare_pre_post_calibration=True)
    
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir or _resolve_latest_run_dir()

    cfg = ADConfig(
        run_dir=run_dir,
        split_seed=int(args.split_seed),
        model_key=str(args.model),
        train_npz=args.train_npz,
        external_npz=args.external_npz,
        predictions_csv=args.predictions,
        tanimoto_threshold=float(args.tanimoto_threshold),
        cosine_threshold=float(args.cosine_threshold),
        strict_similarity=bool(args.strict_similarity),
        base_method=str(args.base_method),
        base_feature_space=str(args.base_feature_space),
        domain_threshold_quantile=float(args.domain_threshold_quantile),
        knn_k=int(args.knn_k),
        leverage_pca_variance=float(args.leverage_pca_variance),
        leverage_pca_components=int(args.leverage_pca_components) if args.leverage_pca_components is not None else None,
        williams_residual_z=float(args.williams_z),
        skip_som=bool(args.skip_som),
        som_rows=int(args.som_rows),
        som_cols=int(args.som_cols),
        som_iterations=int(args.som_iterations),
        output_dir=args.output_dir,
        inplace_update_predictions=bool(args.inplace),
        make_plots=bool(args.make_plots),
        cosine_block_size=int(args.cosine_block_size),
        # New for calibrated framework
        learn_weights=bool(args.learn_weights),
        calibration_method=str(args.calibration_method),
        weight_search_grid=int(args.weight_search_grid),
        # Manual weight configuration
        w1_tanimoto=float(args.w1_tanimoto) if args.w1_tanimoto is not None else None,
        w2_cosine=float(args.w2_cosine) if args.w2_cosine is not None else None,
        w3_similarity=float(args.w3_similarity) if args.w3_similarity is not None else None,
        w4_density=float(args.w4_density) if args.w4_density is not None else None,
        # AD Score power parameter
        ad_score_power=float(args.ad_score_power),
        # Calibration integration parameters
        logit_shrinkage_method=str(args.logit_shrinkage_method),
        detect_calibration=bool(args.detect_calibration),
        compare_pre_post_calibration=bool(args.compare_pre_post_calibration),
    )

    summary = compute_and_export(cfg)
    print("[OK] AD export complete")
    print(f"  - Output dir: {summary['output_dir']}")
    print(f"  - AD table: {summary['exports']['ad_external_predictions_csv']}")
    print(f"  - In-domain rate: {summary['rates']['in_domain']:.1%}")
    print(f"  - AD_Score (mean±std): {summary['continuous_ad_scores']['ad_score_mean']:.3f}±{summary['continuous_ad_scores']['ad_score_std']:.3f}")
    print(f"  - AD_Score range: [{summary['continuous_ad_scores']['ad_score_min']:.3f}, {summary['continuous_ad_scores']['ad_score_max']:.3f}]")
    print(f"  - AD Score Power (exponent k): {summary['continuous_ad_scores']['ad_score_power']:.2f}")
    print(f"  - Final_Score (mean±std): {summary['continuous_ad_scores']['final_score_mean']:.3f}±{summary['continuous_ad_scores']['final_score_std']:.3f}")
    print(f"  - Final_Score_Shrunk (mean): {summary['continuous_ad_scores']['final_score_shrunk_mean']:.3f}")
    print(f"  - AD-Error Correlation: {summary['continuous_ad_scores']['ad_error_correlation']:.4f}")
    
    weights = summary['continuous_ad_scores']['sim_score_weights']
    ad_weights = summary['continuous_ad_scores']['ad_score_weights']
    print(f"  - Sim_Score weights: Tanimoto={weights['tanimoto']:.3f}, Cosine={weights['cosine']:.3f}")
    print(f"  - AD_Score weights: Similarity={ad_weights['similarity']:.3f}, Density={ad_weights['density']:.3f}")
    
    if summary['continuous_ad_scores']['weight_learning_config']:
        print(f"  - Weights learned: {bool(summary['continuous_ad_scores'].get('weights_learned', False))}")
    
    # TASK 4: Print calibration diagnostics
    calib_diag = summary.get('calibration_diagnostics', {})
    if calib_diag:
        print(f"  - Calibration diagnostics:")
        print(f"    - Raw Brier Score: {calib_diag['raw']['brier_score']:.4f}")
        print(f"    - Raw ECE: {calib_diag['raw']['ece']:.4f}")
        if 'calibrated' in calib_diag:
            print(f"    - Calibrated Brier: {calib_diag['calibrated']['brier_score']:.4f}")
            print(f"    - Calibrated ECE: {calib_diag['calibrated']['ece']:.4f}")
        if 'in_domain' in calib_diag:
            print(f"    - In-Domain Brier: {calib_diag['in_domain']['brier_score']:.4f}")
        if 'out_domain' in calib_diag:
            print(f"    - Out-Domain Brier: {calib_diag['out_domain']['brier_score']:.4f}")
    
    # TASK 3: Print shrinkage method info
    calib_info = summary.get('calibration', {})
    print(f"  - Calibration detected: {calib_info.get('detected', False)}")
    print(f"  - Shrinkage method: {calib_info.get('shrinkage_method_used', 'unknown')}")
    print(f"  - Probability source: {summary.get('probability_source', 'unknown')}")
    
    print(f"  - Calibration curve: {summary['exports'].get('ad_calibration_curve_csv', 'N/A')}")

    print("[DEBUG CLI]")
    print("tanimoto_threshold:", args.tanimoto_threshold)
    print("cosine_threshold:", args.cosine_threshold)
    print("strict_similarity:", args.strict_similarity)
    print("domain_threshold_quantile:", args.domain_threshold_quantile)
    

if __name__ == "__main__":
    main()


# %%
# Plotting-only cell (interactive)
##############################################################################
# Goal: visualize AD exports WITHOUT recomputing
# Inputs (from OUT_DIR):
#   - ad_external_predictions.csv (required; source of truth for plotting)
#   - ad_plot_data.npz            (optional; fast arrays)
#   - ad_summary.json             (optional; h_star + config metadata)
#   - ad_calibration_curve.csv    (optional; bin-level curve)
#
# Usage:
#   - Set OUT_DIR below, OR set env var AD_OUT_DIR to an output folder.
#   - Run this cell in VSCode / Jupyter (will not execute in CLI mode).
##############################################################################

from pathlib import Path


def _in_ipython() -> bool:
    try:
        from IPython import get_ipython  # type: ignore
        return get_ipython() is not None
    except Exception:
        return False


if _in_ipython():
    import json
    import os
    from typing import Any, Dict, List, Optional, Tuple

    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator

    # Plot style config
    PLOT_STYLE: Dict[str, Any] = {
        "font_family": "Cambria",
        "font_size": 11,
        "dpi": 600,
        "grid_alpha": 0.25,
        "axes_linewidth": 1.1,
    }

    _C_IN = "#4C72B0"
    _C_OUT = "#DD8452"
    _C_BAR = [_C_IN, _C_OUT]

    _LEGEND_KW: Dict[str, Any] = dict(
        loc="upper right",
        frameon=True,
        fancybox=True,
        edgecolor="0.70",
        framealpha=0.90,
        fontsize=9,
    )

    def _guess_project_root() -> Path:
        candidates: List[Path] = []
        try:
            candidates.append(Path(__file__).resolve().parent.parent)
        except Exception:
            pass
        cwd = Path.cwd().resolve()
        candidates.append(cwd)
        candidates.extend(list(cwd.parents))
        for c in candidates:
            if (c / "models_out").exists():
                return c
        return cwd

    PROJECT_ROOT = _guess_project_root()

    # Adjust here, or set env var AD_OUT_DIR
    OUT_DIR = Path(
        os.environ.get(
            "AD_OUT_DIR",
            str(
                PROJECT_ROOT
                / "models_out/qsar_ml_20260412_162829"
                / "split_seed_12345/validation/applicability_domain/SVC/seed_12345"
            ),
        )
    ).expanduser().resolve()

    def _style_axis(ax) -> None:
        ax.grid(True, linestyle=":", alpha=PLOT_STYLE["grid_alpha"])
        ax.tick_params(direction="out", length=4, width=1)

    def _save(fig, path_base: Path) -> None:
        fig.savefig(path_base.with_suffix(".png"))
        fig.savefig(path_base.with_suffix(".svg"))
        plt.close(fig)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [PLOT_STYLE["font_family"]],
            "font.size": PLOT_STYLE["font_size"],
            "figure.dpi": PLOT_STYLE["dpi"],
            "savefig.dpi": PLOT_STYLE["dpi"],
            "axes.linewidth": PLOT_STYLE["axes_linewidth"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.alpha": PLOT_STYLE["grid_alpha"],
        }
    )

    def _col(df: "pd.DataFrame", names: List[str], *, required: bool = False, dtype=None):
        for name in names:
            if name in df.columns:
                out = df[name].to_numpy(copy=False)
                return out.astype(dtype, copy=False) if dtype is not None else out
        if required:
            raise KeyError(f"Missing required column. Tried: {names}. Available: {list(df.columns)[:30]} ...")
        return None

    def _load_outputs(out_dir: Path) -> Tuple["pd.DataFrame", Optional[dict], Optional[dict], Optional["pd.DataFrame"]]:
        ad_csv = out_dir / "ad_external_predictions.csv"
        if not ad_csv.exists():
            raise FileNotFoundError(f"Missing required file: {ad_csv}")
        df_local = pd.read_csv(ad_csv)

        plot_npz = out_dir / "ad_plot_data.npz"
        plot_data_local = None
        if plot_npz.exists():
            plot_data_local = dict(np.load(plot_npz, allow_pickle=True))

        summary_path = out_dir / "ad_summary.json"
        summary_local = None
        if summary_path.exists():
            summary_local = json.loads(summary_path.read_text())

        cal_curve_path = out_dir / "ad_calibration_curve.csv"
        cal_df_local = pd.read_csv(cal_curve_path) if cal_curve_path.exists() else None
        return df_local, plot_data_local, summary_local, cal_df_local

    df, plot_data, summary_dict, cal_df = _load_outputs(OUT_DIR)
    print(f"[INFO] Plot OUT_DIR: {OUT_DIR}")
    print(f"[INFO] Loaded rows: {len(df)}; cols: {len(df.columns)}")

    in_domain = _col(df, ["In_Domain", "in_domain"], dtype=bool)
    if in_domain is None and plot_data is not None and "in_domain" in plot_data:
        in_domain = np.asarray(plot_data["in_domain"]).astype(bool)
    if in_domain is None:
        in_domain = np.ones((len(df),), dtype=bool)

    leverage = _col(df, ["Leverage", "leverage"])
    if leverage is None and plot_data is not None and "leverage" in plot_data:
        leverage = np.asarray(plot_data["leverage"])
    std_resid = _col(df, ["StdResidual", "std_resid", "DevianceResidual"])
    if std_resid is None and plot_data is not None and "std_resid" in plot_data:
        std_resid = np.asarray(plot_data["std_resid"])

    error = _col(df, ["LogLoss", "log_loss", "AbsProbError", "abs_prob_error", "AbsError", "abs_error"])
    tanimoto = _col(df, ["Tanimoto_max", "tanimoto_max"])
    density_score = _col(df, ["Density_Score", "density_score"])
    if density_score is None and plot_data is not None and "density_score" in plot_data:
        density_score = np.asarray(plot_data["density_score"])
    ad_score = _col(df, ["AD_Score", "ad_score"])
    if ad_score is None and plot_data is not None and "ad_score" in plot_data:
        ad_score = np.asarray(plot_data["ad_score"])

    # h_star: prefer explicit export, then summary, then estimate.
    h_star = None
    h_star_col = _col(df, ["Leverage_h_star", "h_star"])
    if h_star_col is not None:
        uniq = pd.unique(pd.Series(h_star_col).dropna())
        if len(uniq) == 1:
            h_star = float(uniq[0])
    if h_star is None and isinstance(summary_dict, dict) and "h_star" in summary_dict:
        try:
            h_star = float(summary_dict["h_star"])
        except Exception:
            h_star = None
    if h_star is None and leverage is not None and len(leverage) > 0:
        mean_leverage = float(np.mean(leverage))
        n_samples = int(len(leverage))
        p_estimated = max(1, int(mean_leverage * n_samples - 1))
        h_star = float(3.0 * (p_estimated + 1) / max(1, n_samples))

    if plot_data is not None and "leverage" in plot_data and len(plot_data["leverage"]) != len(df):
        print(f"[WARN] npz length mismatch: leverage={len(plot_data['leverage'])} vs csv={len(df)}. Using CSV columns first.")

    # 1) Williams plot (Leverage vs StdResidual)
    if leverage is not None and std_resid is not None:
        fig, ax = plt.subplots(figsize=(4.8, 3.6), constrained_layout=True)
        ax.scatter(
            leverage[in_domain], std_resid[in_domain],
            s=20, alpha=0.55, color=_C_IN, edgecolors="none", label="In-domain",
        )
        ax.scatter(
            leverage[~in_domain], std_resid[~in_domain],
            s=26, alpha=0.80, color=_C_OUT, marker="D", edgecolors="white",
            linewidths=0.4, label="Out-of-domain",
        )
        if h_star is not None:
            ax.axvline(x=float(h_star), color="red", linestyle="--", linewidth=1.5, alpha=0.8, label=f"h* = {h_star:.3f}")
        ax.axhline(y=2, color="purple", linestyle="--", linewidth=1.2, alpha=0.7, label="±2σ")
        ax.axhline(y=-2, color="purple", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.axhline(y=3, color="darkred", linestyle=":", linewidth=1.0, alpha=0.6, label="±3σ")
        ax.axhline(y=-3, color="darkred", linestyle=":", linewidth=1.0, alpha=0.6)
        ax.set_xlabel("Leverage ($h$)")
        ax.set_ylabel("Std. deviance residual")
        ax.set_title("Williams Plot: Leverage vs Residual")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
        ax.set_ylim(-4, 4)
        _style_axis(ax)
        legend_kw = dict(_LEGEND_KW)
        if len(leverage) > 50:
            legend_kw.update({"loc": "upper left", "bbox_to_anchor": (1.02, 1), "borderaxespad": 0.0})
        ax.legend(**legend_kw)
        plt.show()
        _save(fig, OUT_DIR / "ad_vs_residual")
        print("[EXPORT] ad_vs_residual")
    else:
        print("[WARN] Skip Williams plot (missing Leverage/StdResidual)")

    # 2) Density score vs error
    if density_score is not None and error is not None:
        fig, ax = plt.subplots(figsize=(4.4, 3.2), constrained_layout=True)
        ax.scatter(
            density_score[in_domain], error[in_domain],
            s=20, alpha=0.55, color=_C_IN, edgecolors="none", label="In-domain",
        )
        ax.scatter(
            density_score[~in_domain], error[~in_domain],
            s=26, alpha=0.80, color=_C_OUT, marker="D", edgecolors="white",
            linewidths=0.4, label="Out-of-domain",
        )
        ax.set_xlabel("Density Score")
        ax.set_ylabel("Error (LogLoss / AbsProbError)")
        ax.set_title("Density vs Prediction Error")
        _style_axis(ax)
        ax.legend(**_LEGEND_KW)
        plt.show()
        _save(fig, OUT_DIR / "density_vs_error")
        print("[EXPORT] density_vs_error")
    else:
        print("[WARN] Skip Density plot (missing Density_Score or Error column)")

    # 3) Similarity (Tanimoto) vs error
    if tanimoto is not None and error is not None:
        fig, ax = plt.subplots(figsize=(4.4, 3.2), constrained_layout=True)
        ax.scatter(
            tanimoto[in_domain], error[in_domain],
            s=20, alpha=0.55, color=_C_IN, edgecolors="none", label="In-domain",
        )
        ax.scatter(
            tanimoto[~in_domain], error[~in_domain],
            s=26, alpha=0.80, color=_C_OUT, marker="D", edgecolors="white",
            linewidths=0.4, label="Out-of-domain",
        )
        ax.set_xlabel("Max Tanimoto Similarity")
        ax.set_ylabel("Error (LogLoss / AbsProbError)")
        ax.set_title("Fingerprint Similarity vs Prediction Error")
        _style_axis(ax)
        ax.legend(**_LEGEND_KW)
        plt.show()
        _save(fig, OUT_DIR / "similarity_vs_error")
        print("[EXPORT] similarity_vs_error")
    else:
        print("[WARN] Skip Similarity plot (missing Tanimoto_max or Error column)")

    # 4) Coverage bar
    fig, ax = plt.subplots(figsize=(3.6, 3.2), constrained_layout=True)
    n_total = int(len(in_domain))
    counts = [int(np.sum(in_domain)), int(np.sum(~in_domain))]
    labels = ["In-domain", "Out-of-domain"]
    bars = ax.bar(labels, counts, color=_C_BAR, edgecolor="white", linewidth=0.8)
    for bar, cnt in zip(bars, counts):
        pct = 100.0 * cnt / n_total if n_total > 0 else 0.0
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max(counts) * 0.02 if max(counts) > 0 else 0.5,
            f"{cnt} ({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("Number of samples")
    ax.set_title("AD Coverage (In_Domain flag)")
    ax.set_ylim(0, max(counts) * 1.18 if max(counts) > 0 else 1.0)
    _style_axis(ax)
    plt.show()
    _save(fig, OUT_DIR / "ad_coverage")
    print("[EXPORT] ad_coverage")

    # 5) Optional calibration curve + metrics
    if cal_df is not None and isinstance(summary_dict, dict):
        scores_dict = summary_dict.get("continuous_ad_scores", {})
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), constrained_layout=True)
        ax = axes[0]
        if len(cal_df) > 0 and {"ad_mean", "error_mean", "error_std"}.issubset(set(cal_df.columns)):
            ax.errorbar(
                cal_df["ad_mean"],
                cal_df["error_mean"],
                yerr=cal_df["error_std"],
                fmt="o-",
                capsize=4,
                markersize=5,
                color=_C_IN,
                ecolor="0.50",
                label="Mean ± Std",
                linewidth=1.5,
                alpha=0.85,
            )
            ax.fill_between(
                cal_df["ad_mean"],
                cal_df["error_mean"] - cal_df["error_std"],
                cal_df["error_mean"] + cal_df["error_std"],
                color=_C_IN,
                alpha=0.15,
            )
        ax.set_xlabel("AD Score")
        ax.set_ylabel("LogLoss Error")
        ax.set_title("Calibration: AD Score vs Error")
        _style_axis(ax)
        ax.legend(**_LEGEND_KW)

        ax = axes[1]
        ad_error_corr = float(scores_dict.get("ad_error_correlation", 0.0) or 0.0)
        shrunk_mean = float(scores_dict.get("final_score_shrunk_mean", scores_dict.get("final_score_logit_mean", 0.0)) or 0.0)
        metrics_text = (
            f"AD-Error Corr: {ad_error_corr:+.4f}\n"
            f"Density ref median: {scores_dict.get('density_reference_median', 'NA')}\n"
            f"\nAD_Score:\n"
            f"  μ = {scores_dict.get('ad_score_mean', 0):.3f}\n"
            f"  σ = {scores_dict.get('ad_score_std', 0):.3f}\n"
            f"\nFinal_Score:\n"
            f"  μ = {scores_dict.get('final_score_mean', 0):.3f}\n"
            f"\nFinal_Score_Shrunk:\n"
            f"  μ = {shrunk_mean:.3f}"
        )
        ax.text(
            0.08,
            0.95,
            metrics_text,
            transform=ax.transAxes,
            fontsize=9,
            family="monospace",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", edgecolor="0.70", alpha=0.35),
        )
        ax.set_title("Summary Metrics")
        ax.axis("off")
        plt.show()
        _save(fig, OUT_DIR / "ad_calibration_metrics")
        print("[EXPORT] ad_calibration_metrics")
    else:
        print("[INFO] Calibration files not found (skipped calibration plot)")

    # Advanced: error distribution + activity cliffs
    from scipy.stats import gaussian_kde

    if error is not None:
        fig, ax = plt.subplots(figsize=(5.0, 3.6), constrained_layout=True)
        err_in = np.asarray(error)[in_domain]
        err_out = np.asarray(error)[~in_domain]
        if len(err_in) > 1:
            kde_in = gaussian_kde(err_in)
            x_in = np.linspace(float(np.min(err_in)), float(np.max(err_in)), 100)
            ax.plot(x_in, kde_in(x_in), color=_C_IN, linewidth=2, label="In-domain", alpha=0.8)
            ax.fill_between(x_in, kde_in(x_in), alpha=0.3, color=_C_IN)
            ax.axvline(float(np.mean(err_in)), color=_C_IN, linestyle="--", alpha=0.7, linewidth=1)
        if len(err_out) > 1:
            kde_out = gaussian_kde(err_out)
            x_out = np.linspace(float(np.min(err_out)), float(np.max(err_out)), 100)
            ax.plot(x_out, kde_out(x_out), color=_C_OUT, linewidth=2, label="Out-of-domain", alpha=0.8)
            ax.fill_between(x_out, kde_out(x_out), alpha=0.3, color=_C_OUT)
            ax.axvline(float(np.mean(err_out)), color=_C_OUT, linestyle="--", alpha=0.7, linewidth=1)
        ax.set_xlabel("Error (LogLoss / AbsProbError)")
        ax.set_ylabel("Density")
        ax.set_title("Error Distribution: In-domain vs Out-of-domain")
        _style_axis(ax)
        ax.legend(**_LEGEND_KW)
        plt.show()
        _save(fig, OUT_DIR / "error_distribution_comparison")
        print("[EXPORT] error_distribution_comparison")

    if leverage is not None and std_resid is not None and h_star is not None:
        fig, ax = plt.subplots(figsize=(5.2, 3.8), constrained_layout=True)
        activity_cliff_mask = (np.asarray(leverage) < float(h_star)) & (np.abs(np.asarray(std_resid)) > 2)
        ax.scatter(
            np.asarray(leverage)[in_domain & ~activity_cliff_mask],
            np.asarray(std_resid)[in_domain & ~activity_cliff_mask],
            s=18, alpha=0.4, color=_C_IN, edgecolors="none", label="In-domain",
        )
        ax.scatter(
            np.asarray(leverage)[~in_domain & ~activity_cliff_mask],
            np.asarray(std_resid)[~in_domain & ~activity_cliff_mask],
            s=22, alpha=0.6, color=_C_OUT, marker="D", edgecolors="white",
            linewidths=0.4, label="Out-of-domain",
        )
        if np.any(activity_cliff_mask):
            ax.scatter(
                np.asarray(leverage)[activity_cliff_mask],
                np.asarray(std_resid)[activity_cliff_mask],
                s=60, alpha=0.9, color="red", marker="*", edgecolors="darkred",
                linewidths=1.0, label="Activity Cliff", zorder=10
            )
        ax.axvline(x=float(h_star), color="red", linestyle="--", linewidth=1.5, alpha=0.8, label=f"h* = {h_star:.3f}")
        ax.axhline(y=2, color="purple", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.axhline(y=-2, color="purple", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.set_xlabel("Leverage ($h$)")
        ax.set_ylabel("Std. deviance residual")
        ax.set_title(f"Activity Cliff Detection (n={int(np.sum(activity_cliff_mask))})")
        ax.set_ylim(-4, 4)
        _style_axis(ax)
        ax.legend(**{
            "loc": "upper left",
            "bbox_to_anchor": (1.02, 1),
            "borderaxespad": 0.0,
            "frameon": True,
            "fancybox": True,
            "edgecolor": "0.70",
            "framealpha": 0.90,
            "fontsize": 9,
        })
        plt.show()
        _save(fig, OUT_DIR / "activity_cliff_detection")
        print("[EXPORT] activity_cliff_detection")

    print("[OK] AD visualizations complete.")
