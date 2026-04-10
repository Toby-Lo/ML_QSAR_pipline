#!/usr/bin/env python3
"""
Linear-model SHAP interpretation for step10_qsar_ml.py outputs.

Models supported:
  - LR  (Logistic Regression)
  - SVC (sklearn.svm.SVC)

Explainer logic:
  - LR: shap.LinearExplainer
  - SVC:
      - if model.kernel == "linear" -> shap.LinearExplainer
      - else -> shap.KernelExplainer (background downsample via shap.sample, default 100)

This script consumes the SHAP-ready bundles exported by step10 under:
  <run_dir>/split_seed_<N>/data/shap/

and exports (per-model) the same artifacts as step23_interpretations_tree.py:
  - feature_importance.csv
  - shap_values_external.npz
  - shap_meta.json

Structure:
  - Upper half: compute + export
  - Lower half: plotting-only cell(s) (Nature style)

python scripts/step24_interpretations_linear.py \
  --run-dir models_out/qsar_ml_20260409_222051 \
  --split-seed 42
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
    "models": ["LR", "SVC"],
    "task": "classification",
    "max_samples": None,
    "random_state": 42,
    "feature_name_mode": "raw",  # raw | pretty
    "kernel_background_samples": 100,
    "output_dir": None,  # default: <split_seed_dir>/shap_analysis
}


# %%
LINEAR_MODELS = {"LR", "SVC"}


@dataclass
class ShapConfig:
    run_dir: Path
    split_seed: int
    models: List[str]
    task: str = "classification"
    max_samples: Optional[int] = None
    random_state: int = 42
    feature_name_mode: str = "raw"
    kernel_background_samples: int = 100
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
    """Normalize SHAP outputs to shape (n_samples, n_features)."""
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
        proba = model.predict_proba(X)
        proba = np.asarray(proba)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1].astype(np.float64)
        return proba.reshape(-1).astype(np.float64)
    raise AttributeError("Model does not implement predict_proba; KernelExplainer requires probabilistic output here.")


def _sample_background(X_bg: pd.DataFrame, n: int, random_state: int) -> pd.DataFrame:
    target = int(max(1, n))
    if len(X_bg) <= target:
        return X_bg
    if shap is not None and hasattr(shap, "sample"):
        try:
            return shap.sample(X_bg, target, random_state=random_state)
        except TypeError:
            return shap.sample(X_bg, target)
        except Exception:
            pass
    rng = np.random.default_rng(int(random_state))
    idx = rng.choice(len(X_bg), target, replace=False)
    idx = np.asarray(sorted(idx), dtype=int)
    return X_bg.iloc[idx]


def compute_linear_shap_for_model(
    *,
    model_key: str,
    task: str,
    model: Any,
    X_explain: np.ndarray,
    feature_names: List[str],
    background_X: Optional[np.ndarray],
    kernel_background_samples: int,
    random_state: int,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    if shap is None:
        raise ImportError("shap is not installed. Install with `pip install shap` to run SHAP interpretation.")

    X_df = pd.DataFrame(X_explain, columns=feature_names)
    bg_df = pd.DataFrame(background_X, columns=feature_names) if background_X is not None else X_df
    used: Dict[str, Any] = {"method": None, "model_output": None}

    model_key = str(model_key).upper()
    if model_key == "LR":
        explainer = shap.LinearExplainer(model, bg_df, feature_perturbation="interventional")
        raw_values = explainer.shap_values(X_df)
        base_val = explainer.expected_value
        used = {"method": "linear", "model_output": ("probability" if task == "classification" else "raw")}
    elif model_key == "SVC":
        kernel = getattr(model, "kernel", None)
        if str(kernel).lower() == "linear" and hasattr(model, "coef_"):
            explainer = shap.LinearExplainer(model, bg_df, feature_perturbation="interventional")
            raw_values = explainer.shap_values(X_df)
            base_val = explainer.expected_value
            used = {"method": "linear", "model_output": ("probability" if task == "classification" else "raw")}
        else:
            bg_small = _sample_background(bg_df, int(kernel_background_samples), int(random_state))

            def predict_fn(data: np.ndarray) -> np.ndarray:
                arr = np.asarray(data, dtype=float)
                return _predict_proba_class1(model, arr)

            explainer = shap.KernelExplainer(predict_fn, bg_small)
            raw_values = explainer.shap_values(X_df)
            base_val = explainer.expected_value
            used = {"method": "kernel", "model_output": "probability"}
    else:
        raise ValueError(f"Unsupported model for this script: {model_key} (expected LR/SVC)")

    values = _normalize_shap_values(raw_values, task)
    values = _coerce_shap_to_2d(values, n_samples=X_df.shape[0], n_features=len(feature_names), task=task)

    # expected_value could be scalar/list/array; collapse to single scalar.
    if isinstance(base_val, (list, tuple, np.ndarray)):
        base_arr = np.asarray(base_val, dtype=np.float64).reshape(-1)
        if task == "classification" and base_arr.size > 1:
            base_val_out = float(base_arr[1])
        elif base_arr.size:
            base_val_out = float(base_arr[0])
        else:
            base_val_out = float("nan")
    else:
        base_val_out = float(base_val)
    if not np.isfinite(base_val_out):
        try:
            base_val_out = float(np.nanmean(np.asarray(base_val, dtype=np.float64)))
        except Exception:
            base_val_out = float("nan")

    return values.astype(np.float64), base_val_out, used


def compute_and_export(config: ShapConfig) -> Dict[str, Any]:
    split_dir = _resolve_split_dir(config.run_dir, config.split_seed)
    shap_dir = split_dir / "data" / "shap"
    manifest = _load_manifest(shap_dir)

    out_dir = config.output_dir or (split_dir / "shap_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = manifest.get("rows", [])
    row_by_model = {str(r.get("model")).upper(): r for r in rows if r.get("model")}

    model_keys = [m.strip().upper() for m in config.models if m.strip()]
    model_keys = [m for m in model_keys if m in LINEAR_MODELS]
    if not model_keys:
        raise ValueError("No supported models selected. Choose from: LR,SVC")

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
            X_use = X_ext[idx]
            y_use = y_ext[idx]
            ids_use = [ids_ext[i] for i in idx.tolist()]
            smiles_use = [smiles_ext[i] for i in idx.tolist()]
        else:
            X_use, y_use, ids_use, smiles_use = X_ext, y_ext, ids_ext, smiles_ext

        model_path = _find_model_path(split_dir, model_key, config.split_seed)
        model = joblib.load(model_path)

        shap_values, base_value, shap_info = compute_linear_shap_for_model(
            model_key=model_key,
            task=config.task,
            model=model,
            X_explain=X_use,
            feature_names=feature_names,
            background_X=np.asarray(bg["X"], dtype=np.float64),
            kernel_background_samples=int(config.kernel_background_samples),
            random_state=int(config.random_state),
        )

        y_prob = None
        try:
            y_prob = _predict_proba_class1(model, X_use)
        except Exception:
            y_prob = None

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
            X=X_use.astype(np.float32),
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
            "n_external_total": int(len(X_ext)),
            "n_explained": int(len(X_use)),
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
    (out_dir / "linear_shap_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# %%
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LR/SVC SHAP interpretation from step10 SHAP-ready bundles")
    p.add_argument("--run-dir", type=Path, required=True, help="Run directory (models_out/qsar_ml_YYYYMMDD_HHMMSS)")
    p.add_argument("--split-seed", type=int, required=True)
    p.add_argument("--models", default="LR,SVC", help="Comma-separated models")
    p.add_argument("--task", choices=["classification", "regression"], default="classification")
    p.add_argument("--max-samples", type=int, help="Max external samples to explain (random subset)")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--feature-name-mode", choices=["raw", "pretty"], default="raw")
    p.add_argument("--kernel-background-samples", type=int, default=100, help="Background samples for KernelExplainer")
    p.add_argument("--output-dir", type=Path, help="Output directory (default: <split_seed_dir>/shap_analysis)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ShapConfig(
        run_dir=args.run_dir,
        split_seed=int(args.split_seed),
        models=[m.strip() for m in args.models.split(",") if m.strip()],
        task=str(args.task),
        max_samples=(int(args.max_samples) if args.max_samples else None),
        random_state=int(args.random_state),
        feature_name_mode=str(args.feature_name_mode),
        kernel_background_samples=int(args.kernel_background_samples),
        output_dir=args.output_dir,
    )
    summary = compute_and_export(cfg)
    print("[OK] Linear SHAP export complete")
    print(f"  - Output dir: {summary['output_dir']}")
    print(f"  - Models: {', '.join(summary['models'])}")


if __name__ == "__main__":
    main()


# %%
# Plotting-only cell (interactive, Nature style)
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
        "font_family": "Times New Roman", # Cambria
        "font_size": 10,
        "dpi": 600,
        "max_display": 20,
        "heatmap_samples": 64,
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

        # clean axes
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,

        "axes.grid": False,
        "lines.linewidth": 1.2,
    })

    # --- Inputs ---
    OUT_DIR = Path("../models_out/qsar_ml_20260409_222051/split_seed_42/shap_analysis")     # Relative Path
    MODEL_KEY = "SVC"   # "LR" or "SVC"

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
    fig = plt.figure(figsize=(7.2, 5.2))

    shap.summary_plot(
        shap_values,
        X_df,
        feature_names=feature_display,
        max_display=PLOT_STYLE["max_display"],
        show=False,
        cmap="viridis",
    )

    plt.xlabel("SHAP value (impact on model output)")
    plt.title(f"{MODEL_KEY} | SHAP Summary", fontsize=11, pad=10)

    plt.tight_layout()
    _save_fig(fig, "shap_summary_beeswarm")
    plt.show()

    # 2. SHAP Heatmap
    try:
        n_heat = min(PLOT_STYLE["heatmap_samples"], len(X_df))

        # top features by mean |SHAP|
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_idx = np.argsort(mean_abs_shap)[::-1][:15]

        shap_sub = shap_values[:n_heat][:, top_idx]
        X_sub = X_df.iloc[:n_heat, top_idx]
        feature_sub = [feature_display[i] for i in top_idx]

        explanation = shap.Explanation(
            values=shap_sub,
            data=X_sub,
            feature_names=feature_sub,
        )

        fig = plt.figure(figsize=(8, 0.35 * len(feature_sub) + 2))

        shap.plots.heatmap(
            explanation,
            max_display=len(feature_sub),
            show=False,
        )

        plt.title(f"{MODEL_KEY} | SHAP Heatmap", fontsize=11, pad=10)
        plt.gca().tick_params(axis='y', labelsize=8)

        plt.tight_layout()
        _save_fig(fig, "shap_heatmap_clean")
        plt.show()

    except Exception as e:
        print(f"[WARN] Heatmap skipped: {e}")
# %%
