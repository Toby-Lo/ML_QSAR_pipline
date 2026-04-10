#!/usr/bin/env python3
"""Robustness validation for step10 models via Y-scrambling and visual diagnostics.

This script is intentionally organized with ``# %%`` blocks so it can be used
both as a CLI tool and in an interactive editor.

Example:
  python scripts/step21_model_robustness.py \
    --run-dir models_out/qsar_ml_20260409_214751 \
    --split-seed 42 \
    --models LR,RFC,SVC,XGBC,ETC,MLP \
    --n-permutations 200 \
    --input data/test_data_feature_fingerprint.csv
"""

# %%
from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import r2_score, roc_auc_score

from rdkit import Chem
from rdkit.Chem import Descriptors

try:
    from rdkit.Chem.rdMolDescriptors import MorganGenerator

    _MORGAN_GENERATOR_AVAILABLE = True
except ImportError:
    from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect

    _MORGAN_GENERATOR_AVAILABLE = False


# %%
PLOT_CONFIG: Dict[str, Any] = {
    "default_backend": "Agg",
    "interactive_backend": "Qt5Agg",
    "use_interactive": True,
    "display_plots": True,
    "font_family": "Cambria",
    "font_size": 11,
    "title_fontsize": 14,
    "label_fontsize": 12,
    "legend_fontsize": 10,
    "title_fontweight": "semibold",
    "label_fontweight": "normal",
    "line_width": 1.8,
    "grid_alpha": 0.15,
    "grid_color": "#d9d9d9",
    "tick_color": "#4c4c4c",
    "hist_color": "#003366",
    "hist_edgecolor": "black",
    "hist_edgewidth": 0.3,
    "hist_bins": 24,
    "hist_alpha": 1.0,
    "scatter_color": "#9467bd",
    "scatter_alpha": 0.72,
    "scatter_line_color": "#2c2c2c",
    "scatter_line_width": 1.25,
    "scatter_line_style": "-",
    "actual_line_color": "#d62728",
    "dpi": 300,
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "run_dir": Path("models_out/qsar_ml_20260409_214751"),
    "split_seed": 42,
    "models": ["LR", "RFC", "SVC", "XGBC", "ETC", "MLP"],
    "task": "classification",
    "n_permutations": 200,
    "random_state": 42,
    "input_path": None,
    "id_column": "id",
    "smiles_column": "smiles",
    "label_column": "label",
}

NON_TREE_MODELS = {"LR", "SVC", "MLP"}


# %%
def _has_display() -> bool:
    return bool(
        os.environ.get("DISPLAY")
        or os.environ.get("WAYLAND_DISPLAY")
        or os.environ.get("MIR_SOCKET")
    )


def _setup_matplotlib() -> None:
    backend_override = os.environ.get("MATPLOTLIB_BACKEND")
    if backend_override:
        backend = backend_override
    elif PLOT_CONFIG["use_interactive"] and _has_display():
        backend = PLOT_CONFIG["interactive_backend"]
    else:
        backend = PLOT_CONFIG["default_backend"]
    backend_candidates: List[str] = [backend]
    if backend != "Agg":
        backend_candidates.append("Agg")
    for candidate in backend_candidates:
        try:
            matplotlib.use(candidate, force=True)
            plt_mod = importlib.import_module("matplotlib.pyplot")
            fig = plt_mod.figure()
            plt_mod.close(fig)
            return
        except Exception:
            continue
    matplotlib.use("Agg", force=True)


_setup_matplotlib()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import rcParams  # noqa: E402


def _configure_rcparams() -> None:
    rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [PLOT_CONFIG["font_family"]],
            "font.size": PLOT_CONFIG["font_size"],
            "axes.titlesize": PLOT_CONFIG["title_fontsize"],
            "axes.labelsize": PLOT_CONFIG["label_fontsize"],
            "legend.fontsize": PLOT_CONFIG["legend_fontsize"],
            "axes.titleweight": PLOT_CONFIG["title_fontweight"],
            "axes.labelweight": PLOT_CONFIG["label_fontweight"],
            "axes.linewidth": PLOT_CONFIG["line_width"],
            "axes.edgecolor": "#333333",
            "axes.grid": True,
            "grid.alpha": PLOT_CONFIG["grid_alpha"],
            "grid.color": PLOT_CONFIG["grid_color"],
            "grid.linestyle": "-",
            "savefig.dpi": PLOT_CONFIG["dpi"],
            "figure.dpi": PLOT_CONFIG["dpi"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )
    rcParams["axes.spines.right"] = False
    rcParams["axes.spines.top"] = False


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(True, alpha=PLOT_CONFIG["grid_alpha"], color=PLOT_CONFIG["grid_color"], linewidth=0.9)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(1.3)
        spine.set_visible(True)
    ax.tick_params(colors=PLOT_CONFIG["tick_color"], which="both")


def _save_plot(fig: plt.Figure, base_path: Path) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(base_path.with_suffix(".png"), dpi=PLOT_CONFIG["dpi"])
    fig.savefig(base_path.with_suffix(".svg"))
    if PLOT_CONFIG["display_plots"] and not matplotlib.get_backend().lower().startswith("agg"):
        plt.show(block=False)
        plt.pause(0.001)


_configure_rcparams()


# %%
def setup_logger(run_dir: Path) -> logging.Logger:
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("step21_model_robustness")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_dir / "step21_model_robustness.log")
    sh = logging.StreamHandler()
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def detect_existing_fingerprints(df: pd.DataFrame) -> Optional[np.ndarray]:
    candidates: List[Tuple[int, str]] = []
    for col in df.columns:
        if col.startswith("morgan_"):
            suffix = col.split("morgan_", 1)[-1]
            if suffix.isdigit():
                candidates.append((int(suffix), col))
    if len(candidates) < 2048:
        return None
    sorted_cols = [col for _, col in sorted(candidates, key=lambda x: x[0])][:2048]
    return df[sorted_cols].astype(np.float32).to_numpy(dtype=np.float32)


def compute_morgan_fingerprints(smiles_list: List[str], radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    fps: List[np.ndarray] = []
    generator = MorganGenerator(radius=radius, nBits=n_bits) if _MORGAN_GENERATOR_AVAILABLE else None
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            fps.append(np.zeros(n_bits, dtype=np.float32))
            continue
        if _MORGAN_GENERATOR_AVAILABLE and generator is not None:
            fp = generator.GetFingerprintAsBitVect(mol)
        else:
            fp = GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        fps.append(np.array(fp, dtype=np.float32))
    return np.stack(fps, axis=0)


def compute_rdkit_descriptors(smiles_list: List[str], descriptor_names: List[str]) -> np.ndarray:
    funcs = {name: getattr(Descriptors, name) for name in descriptor_names}
    rows = {name: [] for name in descriptor_names}
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        for name, func in funcs.items():
            try:
                val = float(func(mol)) if mol is not None else float("nan")
            except Exception:
                val = float("nan")
            rows[name].append(val)
    return pd.DataFrame(rows).fillna(0.0).astype(np.float32).to_numpy(dtype=np.float32)


def _load_npz_dict(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def _rebuild_split_arrays(
    split_dir: Path,
    input_path: Path,
    id_column: str,
    smiles_column: str,
    label_column: str,
) -> Dict[str, Dict[str, np.ndarray]]:
    df = read_table(input_path)
    split_idx = json.loads((split_dir / "split_indices.json").read_text())
    train_idx = np.asarray(split_idx["train"], dtype=int)
    ext_idx = np.asarray(split_idx["external"], dtype=int)

    if smiles_column not in df.columns or label_column not in df.columns:
        raise ValueError("Input data must include smiles and label columns for fallback rebuild")

    smiles = df[smiles_column].fillna("").astype(str).tolist()
    y_all = pd.to_numeric(df[label_column], errors="coerce").to_numpy(dtype=int)
    ids_all = df[id_column].astype(str).to_numpy(dtype=object) if id_column in df.columns else np.arange(len(df)).astype(str)

    descriptor_names = json.loads((split_dir / "feature_processors" / "descriptor_names.json").read_text())
    fp_all = detect_existing_fingerprints(df)
    if fp_all is None:
        fp_all = compute_morgan_fingerprints(smiles)
    if set(descriptor_names).issubset(df.columns):
        desc_all = df[descriptor_names].astype(np.float32).fillna(0.0).to_numpy(dtype=np.float32)
    else:
        desc_all = compute_rdkit_descriptors(smiles, descriptor_names)

    smiles_arr = np.asarray(smiles, dtype=object)
    return {
        "dev": {
            "fp": fp_all[train_idx].astype(np.float32),
            "desc": desc_all[train_idx].astype(np.float32),
            "y": y_all[train_idx].astype(np.int32),
            "id": ids_all[train_idx],
            "smiles": smiles_arr[train_idx],
        },
        "external": {
            "fp": fp_all[ext_idx].astype(np.float32),
            "desc": desc_all[ext_idx].astype(np.float32),
            "y": y_all[ext_idx].astype(np.int32),
            "id": ids_all[ext_idx],
            "smiles": smiles_arr[ext_idx],
        },
    }


def load_split_data(
    split_dir: Path,
    input_path: Optional[Path],
    id_column: str,
    smiles_column: str,
    label_column: str,
) -> Dict[str, Dict[str, np.ndarray]]:
    data_dir = split_dir / "data" / "splits"
    dev = _load_npz_dict(data_dir / "dev_train.npz")
    ext = _load_npz_dict(data_dir / "external_test.npz")
    need_fallback = not dev or ("y" not in dev) or ("y" not in ext) or ("fp" not in dev) or ("desc" not in dev)

    rebuilt: Dict[str, Dict[str, np.ndarray]] = {}
    if need_fallback:
        if input_path is None:
            raise ValueError(
                "split data lacks required fields (fp/desc/y). Provide --input to rebuild from source table."
            )
        rebuilt = _rebuild_split_arrays(split_dir, input_path, id_column, smiles_column, label_column)

    def _pick(d: Dict[str, np.ndarray], r: Dict[str, np.ndarray], key: str) -> np.ndarray:
        if key in d:
            return d[key]
        return r[key]

    dev_out = {
        "fp": _pick(dev, rebuilt.get("dev", {}), "fp").astype(np.float32),
        "desc": _pick(dev, rebuilt.get("dev", {}), "desc").astype(np.float32),
        "y": _pick(dev, rebuilt.get("dev", {}), "y").astype(np.int32),
        "id": np.asarray(_pick(dev, rebuilt.get("dev", {}), "id"), dtype=object),
        "smiles": np.asarray(_pick(dev, rebuilt.get("dev", {}), "smiles"), dtype=object),
    }
    ext_out = {
        "fp": _pick(ext, rebuilt.get("external", {}), "fp").astype(np.float32),
        "desc": _pick(ext, rebuilt.get("external", {}), "desc").astype(np.float32),
        "y": _pick(ext, rebuilt.get("external", {}), "y").astype(np.int32),
        "id": np.asarray(_pick(ext, rebuilt.get("external", {}), "id"), dtype=object),
        "smiles": np.asarray(_pick(ext, rebuilt.get("external", {}), "smiles"), dtype=object),
    }
    return {"dev": dev_out, "external": ext_out}


# %%
def _get_scores(model, X: np.ndarray, task: str) -> np.ndarray:
    if task == "classification":
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return np.asarray(proba[:, 1], dtype=float)
            return np.asarray(proba.ravel(), dtype=float)
        if hasattr(model, "decision_function"):
            return np.asarray(model.decision_function(X), dtype=float)
        raise AttributeError(f"{type(model).__name__} has neither predict_proba nor decision_function")
    return np.asarray(model.predict(X), dtype=float)


def _metric(y_true: np.ndarray, y_score: np.ndarray, task: str) -> float:
    if task == "classification":
        try:
            return float(roc_auc_score(y_true, y_score))
        except Exception:
            return float("nan")
    try:
        return float(r2_score(y_true, y_score))
    except Exception:
        return float("nan")


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def prepare_model_inputs(
    split_dir: Path,
    model_key: str,
    split_data: Dict[str, Dict[str, np.ndarray]],
    split_seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model_dir = split_dir / "models" / "full_dev" / model_key / f"seed_{split_seed}"
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory missing: {model_dir}")

    mask = np.load(split_dir / "feature_processors" / "fp_mask.npy")
    fp_dev = split_data["dev"]["fp"][:, mask].astype(np.float32)
    fp_ext = split_data["external"]["fp"][:, mask].astype(np.float32)
    desc_dev = split_data["dev"]["desc"].astype(np.float32)
    desc_ext = split_data["external"]["desc"].astype(np.float32)

    scaler_path = model_dir / "scaler.joblib"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        desc_dev = scaler.transform(desc_dev).astype(np.float32)
        desc_ext = scaler.transform(desc_ext).astype(np.float32)

    X_dev = np.concatenate([fp_dev, desc_dev], axis=1).astype(np.float32)
    X_ext = np.concatenate([fp_ext, desc_ext], axis=1).astype(np.float32)
    y_dev = split_data["dev"]["y"].astype(np.int32)
    y_ext = split_data["external"]["y"].astype(np.int32)
    return X_dev, y_dev, X_ext, y_ext


def run_y_scrambling_for_model(
    split_dir: Path,
    model_key: str,
    split_seed: int,
    split_data: Dict[str, Dict[str, np.ndarray]],
    n_permutations: int,
    task: str,
    random_state: int,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    model_path = split_dir / "models" / "full_dev" / model_key / f"seed_{split_seed}" / "model.joblib"
    model = joblib.load(model_path)
    X_dev, y_dev, X_ext, y_ext = prepare_model_inputs(split_dir, model_key, split_data, split_seed)

    actual_score = _metric(y_ext, _get_scores(model, X_ext, task), task)
    rng = np.random.default_rng(random_state + split_seed + len(model_key))
    rows: List[Dict[str, Any]] = []

    for perm_idx in range(1, n_permutations + 1):
        y_perm = rng.permutation(y_dev)
        perm_model = clone(model)
        perm_model.fit(X_dev, y_perm)
        perm_score = _metric(y_ext, _get_scores(perm_model, X_ext, task), task)
        rows.append(
            {
                "permutation_idx": perm_idx,
                "y_corr": _safe_corr(y_dev.astype(float), y_perm.astype(float)),
                "metric": perm_score,
            }
        )

    df = pd.DataFrame(rows)
    vals = df["metric"].to_numpy(dtype=float) if not df.empty else np.array([], dtype=float)
    mean_perm = float(np.nanmean(vals)) if vals.size else float("nan")
    std_perm = float(np.nanstd(vals)) if vals.size else float("nan")
    z_score = float((actual_score - mean_perm) / std_perm) if std_perm and not np.isnan(std_perm) else float("nan")
    p_value = float((1 + np.sum(vals >= actual_score)) / (len(vals) + 1)) if vals.size else float("nan")

    summary = {
        "model": model_key,
        "task": task,
        "n_permutations": int(n_permutations),
        "actual_metric": float(actual_score),
        "perm_metric_mean": mean_perm,
        "perm_metric_std": std_perm,
        "z_score": z_score,
        "p_value": p_value,
    }
    logger.info(
        f"{model_key}: actual={actual_score:.4f}, perm_mean={mean_perm:.4f}, z={z_score:.3f}, p={p_value:.4f}"
    )
    return df, summary


# %%
def plot_metric_histogram(
    model_key: str,
    permutation_metrics: np.ndarray,
    actual_metric: float,
    out_base: Path,
    metric_label: str,
    stat_annotations: Dict[str, float],
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    _style_axis(ax)
    ax.hist(
        permutation_metrics,
        bins=PLOT_CONFIG["hist_bins"],
        color=PLOT_CONFIG["hist_color"],
        alpha=PLOT_CONFIG["hist_alpha"],
        edgecolor=PLOT_CONFIG["hist_edgecolor"],
        linewidth=PLOT_CONFIG["hist_edgewidth"],
    )
    ax.axvline(actual_metric, color=PLOT_CONFIG["actual_line_color"], linewidth=2.0, label="Actual")
    ax.set_title(f"Y-scrambling: {model_key}")
    ax.set_xlabel(f"Permutation {metric_label}")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
    lines = [
        f"Actual: {actual_metric:.3f}",
        f"Perm Mean: {stat_annotations.get('perm_metric_mean', float('nan')):.3f}",
        f"Perm Std: {stat_annotations.get('perm_metric_std', float('nan')):.3f}",
        f"Z-score: {stat_annotations.get('z_score', float('nan')):.3f}",
        f"P-value: {stat_annotations.get('p_value', float('nan')):.4f}",
    ]
    ax.text(
        0.03,
        0.96,
        "\n".join(lines),
        va="top",
        ha="left",
        transform=ax.transAxes,
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "#dddddd", "boxstyle": "round,pad=0.35"},
    )
    _save_plot(fig, out_base)
    plt.close(fig)


def plot_corr_scatter(model_key: str, corr: np.ndarray, metric: np.ndarray, out_base: Path, metric_label: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    _style_axis(ax)
    ax.scatter(corr, metric, color=PLOT_CONFIG["scatter_color"], alpha=PLOT_CONFIG["scatter_alpha"], s=35)
    if len(corr) >= 2:
        coeff = np.polyfit(corr, metric, 1)
        x_line = np.linspace(float(np.min(corr)), float(np.max(corr)), 50)
        y_line = coeff[0] * x_line + coeff[1]
        ax.plot(
            x_line,
            y_line,
            color=PLOT_CONFIG["scatter_line_color"],
            linewidth=PLOT_CONFIG["scatter_line_width"],
            linestyle=PLOT_CONFIG["scatter_line_style"],
        )
    ax.set_title(f"Permutation Correlation vs {metric_label}: {model_key}")
    ax.set_xlabel("corr(y_true_train, y_permuted_train)")
    ax.set_ylabel(f"Permutation {metric_label}")
    _save_plot(fig, out_base)
    plt.close(fig)


# %%
def load_saved_permutation_results(run_dir: Path, split_seed: int, model_key: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    model_dir = run_dir / f"split_seed_{split_seed}" / "robustness" / model_key
    perm_df = pd.read_csv(model_dir / "y_scrambling_permutations.csv")
    summary = json.loads((model_dir / "y_scrambling_summary.json").read_text())
    return perm_df, summary


def replot_model_from_saved(run_dir: Path, split_seed: int, model_key: str, task: str = "classification") -> None:
    """Interactive helper: tweak PLOT_CONFIG then re-render plots from saved CSV/JSON."""
    perm_df, summary = load_saved_permutation_results(run_dir, split_seed, model_key)
    metric_label = "ROC-AUC" if str(task).lower() == "classification" else "R2"
    fig_dir = run_dir / "figures" / "robustness" / f"split_seed_{split_seed}" / model_key
    ensure_dir(fig_dir)
    plot_metric_histogram(
        model_key=model_key,
        permutation_metrics=perm_df["metric"].to_numpy(dtype=float),
        actual_metric=float(summary["actual_metric"]),
        out_base=fig_dir / "y_scrambling_histogram",
        metric_label=metric_label,
        stat_annotations=summary,
    )
    plot_corr_scatter(
        model_key=model_key,
        corr=perm_df["y_corr"].to_numpy(dtype=float),
        metric=perm_df["metric"].to_numpy(dtype=float),
        out_base=fig_dir / "y_scrambling_corr_scatter",
        metric_label=metric_label,
    )


# %%
def run_robustness(config: Dict[str, Any]) -> Dict[str, Any]:
    run_dir = Path(config["run_dir"])
    split_seed = int(config["split_seed"])
    split_dir = run_dir / f"split_seed_{split_seed}"
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    logger = setup_logger(run_dir)
    logger.info(f"Loading split data for split_seed_{split_seed}")

    input_path = config.get("input_path")
    input_path = Path(input_path) if input_path else None
    split_data = load_split_data(
        split_dir=split_dir,
        input_path=input_path,
        id_column=str(config.get("id_column", "id")),
        smiles_column=str(config.get("smiles_column", "smiles")),
        label_column=str(config.get("label_column", "label")),
    )

    out_dir = split_dir / "robustness"
    ensure_dir(out_dir)
    fig_dir = run_dir / "figures" / "robustness" / f"split_seed_{split_seed}"
    ensure_dir(fig_dir)

    task = str(config["task"]).lower()
    metric_label = "ROC-AUC" if task == "classification" else "R2"
    models = list(config["models"])
    summaries: List[Dict[str, Any]] = []

    for idx, model_key in enumerate(models, start=1):
        logger.info(f"[{idx}/{len(models)}] Running y-scrambling for {model_key}")
        perm_df, summary = run_y_scrambling_for_model(
            split_dir=split_dir,
            model_key=model_key,
            split_seed=split_seed,
            split_data=split_data,
            n_permutations=int(config["n_permutations"]),
            task=task,
            random_state=int(config["random_state"]),
            logger=logger,
        )
        model_out_dir = out_dir / model_key
        ensure_dir(model_out_dir)
        perm_df.to_csv(model_out_dir / "y_scrambling_permutations.csv", index=False)
        (model_out_dir / "y_scrambling_summary.json").write_text(json.dumps(summary, indent=2))
        summaries.append(summary)

        model_fig_dir = fig_dir / model_key
        ensure_dir(model_fig_dir)
        plot_metric_histogram(
            model_key=model_key,
            permutation_metrics=perm_df["metric"].to_numpy(dtype=float),
            actual_metric=float(summary["actual_metric"]),
            out_base=model_fig_dir / "y_scrambling_histogram",
            metric_label=metric_label,
            stat_annotations=summary,
        )
        plot_corr_scatter(
            model_key=model_key,
            corr=perm_df["y_corr"].to_numpy(dtype=float),
            metric=perm_df["metric"].to_numpy(dtype=float),
            out_base=model_fig_dir / "y_scrambling_corr_scatter",
            metric_label=metric_label,
        )

    summary_df = pd.DataFrame(summaries).sort_values("model").reset_index(drop=True)
    summary_df.to_csv(out_dir / "y_scrambling_summary.csv", index=False)
    logger.info(f"Saved summary table: {out_dir / 'y_scrambling_summary.csv'}")
    logger.info(f"Saved figures under: {fig_dir}")
    return {"summary": summary_df, "out_dir": out_dir, "fig_dir": fig_dir}


# %%
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Y-scrambling robustness validation for step10 QSAR models")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory from step10")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--models", default="LR,RFC,SVC,XGBC,ETC,MLP", help="Comma-separated model keys")
    parser.add_argument("--task", choices=["classification", "regression"], default="classification")
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--input", type=Path, help="Optional source table for fallback rebuild of split arrays")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--no-display", action="store_true", help="Disable interactive plot display")
    parser.add_argument("--non-interactive", action="store_true", help="Force non-interactive backend")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.no_display:
        PLOT_CONFIG["display_plots"] = False
    if args.non_interactive:
        PLOT_CONFIG["use_interactive"] = False

    config = {
        "run_dir": args.run_dir,
        "split_seed": args.split_seed,
        "models": [m.strip() for m in args.models.split(",") if m.strip()],
        "task": args.task,
        "n_permutations": args.n_permutations,
        "random_state": args.random_state,
        "input_path": args.input,
        "id_column": args.id_column,
        "smiles_column": args.smiles_column,
        "label_column": args.label_column,
    }
    run_robustness(config)


if __name__ == "__main__":
    main()
