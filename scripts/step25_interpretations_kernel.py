#!/usr/bin/env python3
"""
Kernel SHAP interpretation for step10_qsar_ml.py outputs (slow but flexible).

Models supported:
  - MLP  (sklearn.neural_network.MLPClassifier)
  - SVC  (sklearn.svm.SVC)  -> forced KernelExplainer mode

This script uses shap.KernelExplainer to explain P(class=1) on the external set.

IMPORTANT:
  KernelExplainer can be very slow for high-dimensional inputs (e.g., Morgan fingerprints).
  To make it practical for QSAR datasets, this script *always* downsamples:
    - background: shap.sample(..., 50) by default
    - explain set: shap.sample(..., 50) by default
  Increase these numbers only if you are ready for much longer runtimes.

Inputs:
  It consumes the SHAP-ready bundles exported by step10 under:
    <run_dir>/split_seed_<N>/data/shap/

Outputs (per-model) are 100% compatible with step23_interpretations_tree.py NPZ keys:
  - feature_importance.csv
  - shap_values_external.npz
  - shap_meta.json

Structure:
  - Upper half: compute + export
  - Lower half: plotting-only cell(s) (Nature style; Times New Roman; viridis)

python scripts/step25_interpretations_kernel.py \
    --run-dir models_out/qsar_ml_20260409_222051 \
    --split-seed 42 \
    --models MLP,SVC \
    --background-sampling-n 50 \
    --explain-sampling-n 50 \
    --nsamples auto

"""

# %%
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    import pickle

    class JoblibCompat:
        @staticmethod
        def load(path: Path):
            with open(path, "rb") as f:
                return pickle.load(f)

    joblib = JoblibCompat  # type: ignore

try:
    import shap  # type: ignore
except Exception:
    shap = None  # type: ignore


# %%
"""
Config cell (interactive)
-------------------------

Edit this cell when working in an IDE/notebook. CLI execution is controlled by flags.
"""

USER_CONFIG: Dict[str, Any] = {
    "run_dir": Path("models_out/qsar_ml_20260409_222051"),
    "split_seed": 42,
    "models": ["MLP", "SVC"],
    "task": "classification",
    "max_samples": None,  # None => use external bundle size, then apply explain_sampling_n
    "random_state": 42,
    "feature_name_mode": "raw",  # raw | pretty
    "background_sampling_n": 50,
    "explain_sampling_n": 50,
    "nsamples": "auto",  # passed to KernelExplainer.shap_values
    "output_dir": None,  # default: <split_seed_dir>/shap_analysis
}


# %%
KERNEL_MODELS = {"MLP", "SVC"}


@dataclass
class ShapConfig:
    run_dir: Path
    split_seed: int
    models: List[str]
    task: str = "classification"
    max_samples: Optional[int] = None
    random_state: int = 42
    feature_name_mode: str = "raw"
    background_sampling_n: int = 50
    explain_sampling_n: int = 50
    nsamples: Any = "auto"
    output_dir: Optional[Path] = None


def _resolve_split_dir(run_dir: Path, split_seed: int) -> Path:
    split_dir = run_dir / f"split_seed_{int(split_seed)}"
    if not split_dir.exists():
        raise FileNotFoundError(f"split_seed dir not found: {split_dir}")
    return split_dir


def _load_manifest(shap_dir: Path) -> Dict[str, Any]:
    path = shap_dir / "shap_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing shap_manifest.json: {path}")
    return json.loads(path.read_text())


def _load_shap_bundle(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing SHAP bundle: {path}")
    with np.load(path, allow_pickle=True) as data:
        required = {"X", "y", "id", "smiles", "feature_names", "feature_types", "model", "input_mode"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"SHAP bundle missing keys {sorted(missing)}: {path}")
        out = {k: data[k] for k in data.files}
    return out


def _find_model_path(split_dir: Path, model_key: str, split_seed: int) -> Path:
    model_path = split_dir / "models" / "full_dev" / str(model_key) / f"seed_{int(split_seed)}" / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return model_path


def _format_feature_name(name: str) -> str:
    normalized = str(name).replace("_", " ").strip()
    if not normalized:
        return ""
    words = normalized.split()
    return " ".join(word.capitalize() for word in words)


def _format_feature_names(names: List[str], mode: str) -> List[str]:
    mode = str(mode).strip().lower()
    if mode == "raw":
        return names
    if mode == "pretty":
        return [_format_feature_name(n) for n in names]
    raise ValueError(f"Unknown feature_name_mode: {mode!r} (expected: raw/pretty)")


def _normalize_shap_values(values: Any, task: str) -> np.ndarray:
    if isinstance(values, list) and len(values) > 1 and task == "classification":
        return np.asarray(values[1])
    if hasattr(values, "values"):
        return np.asarray(values.values)
    return np.asarray(values)


def _coerce_shap_to_2d(values: np.ndarray, *, n_samples: int, n_features: int, task: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 3:
        if task == "classification":
            if arr.shape[-1] == 2:
                arr = arr[..., 1]
            elif arr.shape[0] == 2:
                arr = arr[1, ...]
            elif arr.shape[1] == 2:
                arr = arr[:, 1, :]
            else:
                arr = arr[..., 0]
        else:
            arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Unsupported SHAP array ndim={arr.ndim}; shape={arr.shape}")
    if arr.shape == (n_samples, n_features):
        return arr
    if arr.shape == (n_features, n_samples):
        return arr.T
    raise ValueError(f"Unexpected SHAP array shape {arr.shape} (expected {(n_samples, n_features)})")


def _predict_proba_class1(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X))
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1].astype(np.float64)
        return proba.reshape(-1).astype(np.float64)
    raise AttributeError("Model does not implement predict_proba; KernelExplainer requires probabilistic output.")


def _shap_sample_df(df: pd.DataFrame, n: int, random_state: int) -> pd.DataFrame:
    target = int(max(1, n))
    if len(df) <= target:
        return df
    if shap is not None and hasattr(shap, "sample"):
        try:
            return shap.sample(df, target, random_state=random_state)
        except TypeError:
            return shap.sample(df, target)
        except Exception:
            pass
    rng = np.random.default_rng(int(random_state))
    idx = rng.choice(len(df), target, replace=False)
    idx = np.asarray(sorted(idx), dtype=int)
    return df.iloc[idx]


def compute_kernel_shap_for_model(
    *,
    model_key: str,
    task: str,
    model: Any,
    X_explain: np.ndarray,
    feature_names: List[str],
    background_X: np.ndarray,
    background_sampling_n: int,
    explain_sampling_n: int,
    nsamples: Any,
    random_state: int,
) -> Tuple[np.ndarray, float, np.ndarray, Dict[str, Any]]:
    if shap is None:
        raise ImportError("shap is not installed. Install with `pip install shap` to run Kernel SHAP.")
    if task != "classification":
        raise ValueError("This script currently supports classification only (P(class=1) explanations).")

    X_df_full = pd.DataFrame(X_explain, columns=feature_names)
    bg_df_full = pd.DataFrame(background_X, columns=feature_names)

    bg_df = _shap_sample_df(bg_df_full, int(background_sampling_n), int(random_state))
    X_df = _shap_sample_df(X_df_full, int(explain_sampling_n), int(random_state + 1))

    def predict_fn(data: np.ndarray) -> np.ndarray:
        arr = np.asarray(data, dtype=float)
        return _predict_proba_class1(model, arr)

    explainer = shap.KernelExplainer(predict_fn, bg_df)
    raw_values = explainer.shap_values(X_df, nsamples=nsamples)
    base_val = explainer.expected_value

    values = _normalize_shap_values(raw_values, task)
    values = _coerce_shap_to_2d(values, n_samples=X_df.shape[0], n_features=len(feature_names), task=task)

    # expected_value could be scalar/list/array; collapse to single scalar.
    if isinstance(base_val, (list, tuple, np.ndarray)):
        base_arr = np.asarray(base_val, dtype=np.float64).reshape(-1)
        base_val_out = float(base_arr[1] if base_arr.size > 1 else (base_arr[0] if base_arr.size else np.nan))
    else:
        base_val_out = float(base_val)
    if not np.isfinite(base_val_out):
        try:
            base_val_out = float(np.nanmean(np.asarray(base_val, dtype=np.float64)))
        except Exception:
            base_val_out = float("nan")

    used = {
        "method": "kernel",
        "model_output": "probability",
        "background_n": int(len(bg_df)),
        "explain_n": int(len(X_df)),
        "nsamples": nsamples,
    }
    return values.astype(np.float64), base_val_out, X_df.to_numpy(dtype=np.float64), used


def compute_and_export(config: ShapConfig) -> Dict[str, Any]:
    split_dir = _resolve_split_dir(config.run_dir, config.split_seed)
    shap_dir = split_dir / "data" / "shap"
    manifest = _load_manifest(shap_dir)

    out_dir = config.output_dir or (split_dir / "shap_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = manifest.get("rows", [])
    row_by_model = {str(r.get("model")).upper(): r for r in rows if r.get("model")}

    model_keys = [m.strip().upper() for m in config.models if m.strip()]
    model_keys = [m for m in model_keys if m in KERNEL_MODELS]
    if not model_keys:
        raise ValueError("No supported models selected. Choose from: MLP,SVC")

    exports: List[Dict[str, Any]] = []
    rng = np.random.default_rng(int(config.random_state))

    for model_key in model_keys:
        if model_key not in row_by_model:
            raise FileNotFoundError(f"Model {model_key} not found in shap_manifest.json under {shap_dir}")
        row = row_by_model[model_key]

        bg_path = Path(row["background_path"])
        ext_path = Path(row["explain_external_path"])
        bg = _load_shap_bundle(bg_path)
        ext = _load_shap_bundle(ext_path)

        feature_names = [str(x) for x in np.atleast_1d(ext["feature_names"]).tolist()]
        display_feature_names = _format_feature_names(feature_names, config.feature_name_mode)
        feature_types = [str(x) for x in np.atleast_1d(ext["feature_types"]).tolist()]

        X_ext = np.asarray(ext["X"], dtype=np.float64)
        y_ext = np.asarray(ext["y"], dtype=np.float64).reshape(-1)
        ids_ext = [str(x) for x in np.atleast_1d(ext["id"]).tolist()]
        smiles_ext = [str(x) for x in np.atleast_1d(ext["smiles"]).tolist()]

        if config.max_samples is not None and int(config.max_samples) > 0 and len(X_ext) > int(config.max_samples):
            idx = rng.choice(len(X_ext), int(config.max_samples), replace=False)
            idx = np.asarray(sorted(idx), dtype=int)
            X_use_full = X_ext[idx]
            y_use_full = y_ext[idx]
            ids_use_full = [ids_ext[i] for i in idx.tolist()]
            smiles_use_full = [smiles_ext[i] for i in idx.tolist()]
        else:
            X_use_full, y_use_full, ids_use_full, smiles_use_full = X_ext, y_ext, ids_ext, smiles_ext

        model_path = _find_model_path(split_dir, model_key, config.split_seed)
        model = joblib.load(model_path)

        shap_values, base_value, X_used, shap_info = compute_kernel_shap_for_model(
            model_key=model_key,
            task=config.task,
            model=model,
            X_explain=X_use_full,
            feature_names=feature_names,
            background_X=np.asarray(bg["X"], dtype=np.float64),
            background_sampling_n=int(config.background_sampling_n),
            explain_sampling_n=int(config.explain_sampling_n),
            nsamples=config.nsamples,
            random_state=int(config.random_state),
        )

        # NOTE: we export y_prob for the explained (sampled) set only.
        y_prob = None
        try:
            y_prob = _predict_proba_class1(model, X_used)
        except Exception:
            y_prob = None

        # Align ids/smiles/y_true to the sampled explain set.
        # We recover indices by matching rows in X_use_full -> X_used (safe because explain sampling is from X_use_full).
        # For robustness, we just re-sample ids/smiles/y using the same shap.sample logic on a DataFrame.
        explain_df = pd.DataFrame({"id": ids_use_full, "smiles": smiles_use_full, "y_true": y_use_full})
        explain_df_s = _shap_sample_df(explain_df, int(config.explain_sampling_n), int(config.random_state + 1))
        ids_use = explain_df_s["id"].astype(str).tolist()
        smiles_use = explain_df_s["smiles"].astype(str).tolist()
        y_use = explain_df_s["y_true"].astype(float).to_numpy()

        importance = np.abs(shap_values).mean(axis=0)
        imp_df = pd.DataFrame(
            {
                "feature": feature_names,
                "feature_display": display_feature_names,
                "feature_type": feature_types,
                "mean_abs_shap": importance.astype(np.float64),
            }
        ).sort_values("mean_abs_shap", ascending=False)

        model_out_dir = out_dir / model_key
        model_out_dir.mkdir(parents=True, exist_ok=True)

        imp_csv = model_out_dir / "feature_importance.csv"
        imp_df.to_csv(imp_csv, index=False)

        shap_npz = model_out_dir / "shap_values_external.npz"
        np.savez_compressed(
            shap_npz,
            shap_values=shap_values.astype(np.float32),
            base_value=np.asarray([base_value], dtype=np.float64),
            X=X_used.astype(np.float32),
            y_true=y_use.astype(np.float32),
            y_prob=(y_prob.astype(np.float32) if y_prob is not None else np.asarray([], dtype=np.float32)),
            id=np.asarray(ids_use, dtype=object),
            smiles=np.asarray(smiles_use, dtype=object),
            feature_names=np.asarray(feature_names, dtype=object),
            feature_display=np.asarray(display_feature_names, dtype=object),
            feature_types=np.asarray(feature_types, dtype=object),
            model=np.asarray([model_key], dtype=object),
            task=np.asarray([config.task], dtype=object),
        )

        meta = {
            "model": model_key,
            "task": config.task,
            "split_seed": int(config.split_seed),
            "input_mode": str(row.get("input_mode", "")),
            "shap_method": str(shap_info.get("method")),
            "shap_model_output": str(shap_info.get("model_output")),
            "kernel_background_n": int(shap_info.get("background_n", 0)),
            "kernel_explain_n": int(shap_info.get("explain_n", 0)),
            "kernel_nsamples": str(shap_info.get("nsamples")),
            "n_external_total": int(len(X_ext)),
            "n_explained": int(len(X_used)),
            "n_features": int(len(feature_names)),
            "base_value": float(base_value),
            "model_path": str(model_path),
            "background_path": str(bg_path),
            "explain_external_path": str(ext_path),
            "exports": {
                "feature_importance_csv": str(imp_csv),
                "shap_values_external_npz": str(shap_npz),
            },
        }
        (model_out_dir / "shap_meta.json").write_text(json.dumps(meta, indent=2))
        exports.append(meta)

    summary = {
        "run_dir": str(config.run_dir),
        "split_seed": int(config.split_seed),
        "task": config.task,
        "models": model_keys,
        "output_dir": str(out_dir),
        "exports": exports,
    }
    (out_dir / "kernel_shap_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# %%
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MLP/SVC Kernel SHAP interpretation from step10 SHAP-ready bundles")
    p.add_argument("--run-dir", type=Path, required=True, help="Run directory (models_out/qsar_ml_YYYYMMDD_HHMMSS)")
    p.add_argument("--split-seed", type=int, required=True)
    p.add_argument("--models", default="MLP,SVC", help="Comma-separated models (Kernel mode)")
    p.add_argument("--task", choices=["classification"], default="classification")
    p.add_argument("--max-samples", type=int, help="Max external samples to consider before explain sampling")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--feature-name-mode", choices=["raw", "pretty"], default="raw")
    p.add_argument("--background-sampling-n", type=int, default=50, help="Background samples for KernelExplainer")
    p.add_argument("--explain-sampling-n", type=int, default=50, help="Explained samples for KernelExplainer")
    p.add_argument("--nsamples", default="auto", help="nsamples for KernelExplainer.shap_values (auto or int)")
    p.add_argument("--output-dir", type=Path, help="Output directory (default: <split_seed_dir>/shap_analysis)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    nsamples: Any
    if str(args.nsamples).strip().lower() == "auto":
        nsamples = "auto"
    else:
        try:
            nsamples = int(args.nsamples)
        except Exception:
            nsamples = args.nsamples

    cfg = ShapConfig(
        run_dir=args.run_dir,
        split_seed=int(args.split_seed),
        models=[m.strip() for m in args.models.split(",") if m.strip()],
        task=str(args.task),
        max_samples=(int(args.max_samples) if args.max_samples else None),
        random_state=int(args.random_state),
        feature_name_mode=str(args.feature_name_mode),
        background_sampling_n=int(args.background_sampling_n),
        explain_sampling_n=int(args.explain_sampling_n),
        nsamples=nsamples,
        output_dir=args.output_dir,
    )
    summary = compute_and_export(cfg)
    print("[OK] Kernel SHAP export complete")
    print(f"  - Output dir: {summary['output_dir']}")
    print(f"  - Models: {', '.join(summary['models'])}")


if __name__ == "__main__":
    main()


# %%
# Plotting-only cell (interactive, Nature style; viridis; top-15 features heatmap)
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

try:
    import shap
except ImportError:
    shap = None

try:
    from IPython import get_ipython  # type: ignore
    _IN_IPYTHON = get_ipython() is not None
except Exception:
    _IN_IPYTHON = False


if _IN_IPYTHON:
    from matplotlib import pyplot as plt
    import matplotlib as mpl

    PLOT_STYLE: Dict[str, Any] = {
        "font_family": "Times New Roman",
        "font_size": 10,
        "dpi": 600,
        "max_display": 20,
        "heatmap_samples": 64,
        "heatmap_top_features": 15,
        "cmap": "viridis",
    }

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": [PLOT_STYLE["font_family"]],
        "font.size": PLOT_STYLE["font_size"],
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": PLOT_STYLE["dpi"],
        "savefig.dpi": PLOT_STYLE["dpi"],

        # clean Nature-style axes
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.grid": False,
        "lines.linewidth": 1.2,
    })

    # Inputs
    OUT_DIR = Path("../models_out/qsar_ml_20260409_222051/split_seed_42/shap_analysis")
    MODEL_KEY = "MLP"

    npz_path = OUT_DIR / MODEL_KEY / "shap_values_external.npz"

    if not npz_path.exists():
        raise FileNotFoundError(f"Missing SHAP exports: {npz_path}")

    if shap is None:
        raise ImportError("Install SHAP: pip install shap")

    # Load data
    with np.load(npz_path, allow_pickle=True) as data:
        shap_values = np.asarray(data["shap_values"], dtype=np.float64)
        X = np.asarray(data["X"], dtype=np.float64)
        feature_display = [str(x) for x in np.atleast_1d(data["feature_display"]).tolist()]

    X_df = pd.DataFrame(X, columns=feature_display)

    def _save_fig(fig, name: str):
        out_dir = OUT_DIR / MODEL_KEY
        out_dir.mkdir(parents=True, exist_ok=True)

        fig.savefig(out_dir / f"{name}.png", bbox_inches="tight", dpi=PLOT_STYLE["dpi"])
        fig.savefig(out_dir / f"{name}.svg", bbox_inches="tight")

    # 1. SHAP Beeswarm
    fig = plt.figure(figsize=(8, 6))

    shap.summary_plot(
        shap_values,
        X_df,
        feature_names=feature_display,
        max_display=PLOT_STYLE["max_display"],
        show=False,
        cmap=PLOT_STYLE["cmap"],
    )

    plt.title(f"Kernel SHAP Beeswarm | {MODEL_KEY}", fontsize=11, pad=10)
    plt.tight_layout()
    _save_fig(fig, "shap_beeswarm")
    plt.show()

    # 2. SHAP Heatmap (Top-K features)
    try:
        n_heat = min(PLOT_STYLE["heatmap_samples"], len(X_df))

        importance = np.abs(shap_values).mean(axis=0)
        topk = int(PLOT_STYLE["heatmap_top_features"])
        top_idx = np.argsort(-importance)[:topk]

        feat_top = [feature_display[i] for i in top_idx.tolist()]

        shap_sub = shap_values[:n_heat, :][:, top_idx]
        X_sub = X_df.iloc[:n_heat, :][feat_top]

        explanation = shap.Explanation(
            values=shap_sub,
            data=X_sub,
            feature_names=feat_top,
        )

        fig = plt.figure(figsize=(8, 0.35 * len(feat_top) + 2))

        shap.plots.heatmap(
            explanation,
            max_display=topk,
            show=False,
        )

        plt.title(f"Kernel SHAP Heatmap (Top {topk}) | {MODEL_KEY}", fontsize=11, pad=10)
        plt.gca().tick_params(axis='y', labelsize=8)

        plt.tight_layout()
        _save_fig(fig, "shap_heatmap_topk")
        plt.show()

    except Exception as e:
        print(f"[WARN] Heatmap skipped: {e}")
# %%