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
  python scripts/step22_applicability_domain.py \
    --split-seed 12345  --model SVC  --run-dir models_out/qsar_ml_20260410_124055

Usage (interactive):
  1) Run the compute/export block (CLI or Jupyter cell)
  2) Then uncomment the plotting cells below to iterate on visualizations without recomputing
"""

# %%
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# %%
# Heavy deps are imported here so the upper section can be edited easily.
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors

try:
    from minisom import MiniSom  # type: ignore
except Exception:
    MiniSom = None  # type: ignore


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
    williams_residual_z: float = 3.0

    # Base AD method (applied in a chosen feature space)
    # - "leverage": PCA leverage (Williams-style)
    # - "mahalanobis": robust Mahalanobis distance (Ledoit-Wolf shrinkage) with train-quantile threshold
    # - "knn_density": mean distance to kNN with train-quantile threshold (cheap SOM alternative)
    base_method: str = "knn_density"
    base_feature_space: str = "desc"  # "desc" | "full" (full = fp + desc)
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


# %%
"""
Config cell (edit this first)
-----------------------------

Defaults match the requested CLI-like settings:
  --tanimoto-threshold 0.6
  --cosine-threshold 0.7
  --no-strict-similarity  (i.e., strict_similarity=False => leverage OR similarity)
"""

USER_CONFIG: Dict[str, Any] = {
    # If None => auto-pick latest under models_out/qsar_ml_*
    "run_dir": None,
    "split_seed": 42,
    "model_key": "SVC",

    "tanimoto_threshold": 0.60,
    "cosine_threshold": 0.70,
    "strict_similarity": False,

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


def _compute_leverage_pca(X_train_scaled: np.ndarray, X_query_scaled: np.ndarray, variance_ratio: float) -> Tuple[np.ndarray, float, int]:
    """Compute leverage in a PCA subspace explaining `variance_ratio` of the variance."""
    pca = PCA(n_components=float(variance_ratio), svd_solver="full")
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_query_pca = pca.transform(X_query_scaled)

    XtX = X_train_pca.T @ X_train_pca
    inv = np.linalg.pinv(XtX)
    h_query = np.einsum("ij,jk,ik->i", X_query_pca, inv, X_query_pca)

    p = int(X_train_pca.shape[1])
    n = int(X_train_pca.shape[0])
    h_star = (3.0 * p) / n if n > 0 else float("inf")
    return h_query.astype(np.float64), float(h_star), p


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

    if merged.isna().any().any():
        missing = merged[merged.isna().any(axis=1)]
        raise ValueError(
            "Failed to align predictions to external set. "
            f"Missing rows after merge: {len(missing)}/{len(merged)}. "
            "Check whether NPZ ordering matches predictions, or ensure 'id' is consistent."
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
def compute_and_export(config: ADConfig) -> Dict[str, Any]:
    run_dir = config.run_dir
    split_seed = int(config.split_seed)
    split_dir = run_dir / f"split_seed_{split_seed}"

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

    # Descriptor space is used for cosine similarity (to avoid redundancy with fingerprint-based Tanimoto).
    desc_scaler = StandardScaler()
    desc_train_scaled = desc_scaler.fit_transform(desc_train)
    desc_ext_scaled = desc_scaler.transform(desc_ext)

    # Base-domain feature space (for leverage / mahalanobis / kNN density / SOM).
    base_space = str(config.base_feature_space).strip().lower()
    if base_space == "desc":
        X_train_base = desc_train
        X_ext_base = desc_ext
    elif base_space == "full":
        X_train_base = np.concatenate([fp_train, desc_train], axis=1)
        X_ext_base = np.concatenate([fp_ext, desc_ext], axis=1)
    else:
        raise ValueError(f"Unknown base_feature_space: {config.base_feature_space!r} (expected: desc/full)")

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

    if base_method == "leverage":
        leverage, h_star, pca_components = _compute_leverage_pca(
            X_train_scaled=X_train_base_scaled,
            X_query_scaled=X_ext_base_scaled,
            variance_ratio=config.leverage_pca_variance,
        )
        base_in_domain = leverage <= h_star
    elif base_method == "mahalanobis":
        d_train, d_ext = _mahalanobis_distance(X_train_base_scaled, X_ext_base_scaled)
        mahalanobis_dist = d_ext
        mahalanobis_thr = _quantile_threshold(d_train, config.domain_threshold_quantile)
        base_in_domain = mahalanobis_dist <= mahalanobis_thr
    elif base_method == "knn_density":
        d_train, d_ext = _knn_mean_distance(X_train_base_scaled, X_ext_base_scaled, k=config.knn_k)
        knn_mean_dist = d_ext
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

    # Export per-sample AD table.
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

    ad_table_path = output_dir / "ad_external_predictions.csv"
    out_df.to_csv(ad_table_path, index=False)

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
        "exports": {
            "ad_external_predictions_csv": str(ad_table_path),
        },
    }

    (output_dir / "ad_summary.json").write_text(json.dumps(summary, indent=2))

    # Store a compact npz so plotting cells can load without redoing heavy work.
    np.savez_compressed(
        output_dir / "ad_plot_data.npz",
        leverage=leverage,
        std_resid=std_resid,
        in_domain=in_domain.astype(np.int8),
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
            winners_tr = np.array([som.winner(x) for x in X_train_scaled])
            winners_te = np.array([som.winner(x) for x in X_ext_scaled])
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
        choices=["desc", "full"],
        default="desc",
        help='Feature space for base AD method: "desc" uses descriptors only; "full" uses fp+desc.',
    )
    p.add_argument(
        "--domain-threshold-quantile",
        type=float,
        default=0.99,
        help="Train-set quantile used as threshold for Mahalanobis/kNN density base methods.",
    )
    p.add_argument("--knn-k", type=int, default=5, help="k for kNN density base method")

    p.add_argument("--leverage-pca-variance", type=float, default=0.95)
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
        williams_residual_z=float(args.williams_z),
        skip_som=bool(args.skip_som),
        som_rows=int(args.som_rows),
        som_cols=int(args.som_cols),
        som_iterations=int(args.som_iterations),
        output_dir=args.output_dir,
        inplace_update_predictions=bool(args.inplace),
        make_plots=bool(args.make_plots),
        cosine_block_size=int(args.cosine_block_size),
    )

    summary = compute_and_export(cfg)
    print("[OK] AD export complete")
    print(f"  - Output dir: {summary['output_dir']}")
    print(f"  - AD table: {summary['exports']['ad_external_predictions_csv']}")
    print(f"  - In-domain rate: {summary['rates']['in_domain']:.1%}")


if __name__ == "__main__":
    main()


# %%
# Plotting-only cell (interactive)
##############
# Goal:
#   - Quickly visualize AD results WITHOUT recomputing
#   - Export publication-ready figures (PNG + SVG)
#
# Usage:
#   - Modify OUT_DIR below
#   - Run this cell in VSCode / Jupyter
#
# NOTE:
#   - Will NOT execute in CLI mode

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _in_ipython() -> bool:
    try:
        from IPython import get_ipython  # type: ignore
        return get_ipython() is not None
    except Exception:
        return False


if _in_ipython():

    # Plot style config
    PLOT_STYLE: Dict[str, Any] = {
        "font_family": "Cambria", # Cambria, Times New Roman
        "font_size": 11,
        "dpi": 600,
        "grid_alpha": 0.25,
        "axes_linewidth": 1.1,
    }


    # Resolve project root
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


    # 👉 adjust here 
    OUT_DIR = PROJECT_ROOT / "./models_out/qsar_ml_20260410_124055/split_seed_12345/validation/applicability_domain/SVC/seed_12345"

    # Load data
    ad_csv = OUT_DIR / "ad_external_predictions.csv"
    ad_npz = OUT_DIR / "ad_plot_data.npz"

    if not ad_csv.exists() or not ad_npz.exists():
        raise FileNotFoundError(
            "AD export files not found.\n"
            f"CWD: {Path.cwd().resolve()}\n"
            f"PROJECT_ROOT: {PROJECT_ROOT}\n"
            f"Expected:\n  - {ad_csv}\n  - {ad_npz}"
        )

    df = pd.read_csv(ad_csv)
    plot_data = np.load(ad_npz, allow_pickle=True)

    leverage = plot_data["leverage"]
    std_resid = plot_data["std_resid"]
    in_domain = plot_data["in_domain"].astype(bool)


    # Resolve optional columns
    def _safe_get(col_names):
        for c in col_names:
            if c in df.columns:
                return df[c].values
        return None

    density = _safe_get(["KNN_Density", "knn_density"])
    tanimoto = _safe_get(["Tanimoto_max", "tanimoto_max"])
    error = _safe_get(["LogLoss", "logloss", "AbsProbError", "abs_error"])

    # Matplotlib global style
    from matplotlib import pyplot as plt

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

    # Save helper
    def _save(fig, path_base: Path):
        fig.savefig(path_base.with_suffix(".png"))
        fig.savefig(path_base.with_suffix(".svg"))
        plt.close(fig)

    print(f"[INFO] Saving figures to: {OUT_DIR}")

    # 1. Williams plot
    fig, ax = plt.subplots(figsize=(4.4, 3.2))

    ax.scatter(leverage[in_domain], std_resid[in_domain],
               s=18, alpha=0.55, label="In-domain")

    ax.scatter(leverage[~in_domain], std_resid[~in_domain],
               s=18, alpha=0.75, label="Out-of-domain")

    ax.set_xlabel("Leverage (h)")
    ax.set_ylabel("Standardized residual")
    ax.set_title("AD vs Residual")
    ax.legend(frameon=False)

    plt.tight_layout()
    _save(fig, OUT_DIR / "ad_vs_residual")

    # 2. Density vs Error
    if density is not None and error is not None:

        fig, ax = plt.subplots(figsize=(4.4, 3.2))

        ax.scatter(density[in_domain], error[in_domain],
                   s=18, alpha=0.55, label="In-domain")

        ax.scatter(density[~in_domain], error[~in_domain],
                   s=18, alpha=0.75, label="Out-of-domain")

        ax.set_xlabel("KNN Density")
        ax.set_ylabel("Error")
        ax.set_title("Density vs Error")
        ax.legend(frameon=False)

        plt.tight_layout()
        _save(fig, OUT_DIR / "density_vs_error")

    else:
        print("[WARN] Skip Density plot (missing column)")

    # 3. Similarity vs Error
    if tanimoto is not None and error is not None:

        fig, ax = plt.subplots(figsize=(4.4, 3.2))

        ax.scatter(tanimoto[in_domain], error[in_domain],
                   s=18, alpha=0.55, label="In-domain")

        ax.scatter(tanimoto[~in_domain], error[~in_domain],
                   s=18, alpha=0.75, label="Out-of-domain")

        ax.set_xlabel("Tanimoto Similarity")
        ax.set_ylabel("Error")
        ax.set_title("Similarity vs Error")
        ax.legend(frameon=False)

        plt.tight_layout()
        _save(fig, OUT_DIR / "similarity_vs_error")

    else:
        print("[WARN] Skip Similarity plot (missing column)")

    # 4. AD Coverage
    fig, ax = plt.subplots(figsize=(3.5, 3.2))

    counts = [np.sum(in_domain), np.sum(~in_domain)]

    ax.bar(["In-domain", "Out-of-domain"], counts)
    ax.set_ylabel("Number of samples")
    ax.set_title("AD Coverage")

    plt.tight_layout()
    _save(fig, OUT_DIR / "ad_coverage")

    # Final show (optional)
    plt.show()

    print("[DONE] All plots generated.")
# %%
