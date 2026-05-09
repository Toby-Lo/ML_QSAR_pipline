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
  --run-dir models_out/qsar_ml_20260410_124055 \
  --split-seed 12345

python scripts/step24_interpretations_linear.py \
  --run-dir models_out/qsar_ml_20260412_162829 \
  --split-seed 12345 \
  --models SVC \
  --fp-top-k 20

Plots only
python scripts/step24_interpretations_linear.py \
  --run-dir models_out/qsar_ml_20260412_162829 \
  --split-seed 12345 \
  --plot-only \
  --plot-model SVC

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
from rdkit import Chem
from rdkit.Chem import AllChem

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
LINEAR_MODELS = {"SVC"}  ### "LR", "SVC"


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
    fp_top_k: int = 20
    fp_radius: int = 2
    fp_nbits: int = 2048


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


def _parse_fp_bit(feature_name: str) -> Optional[int]:
    name = str(feature_name).strip()
    if name.startswith("fp_"):
        token = name.split("fp_", 1)[1]
        if token.isdigit():
            return int(token)
    return None


def _bit_motifs_from_smiles(smiles: str, bit_id: int, radius: int, nbits: int) -> List[str]:
    if not smiles:
        return []
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    bit_info: Dict[int, List[Tuple[int, int]]] = {}
    try:
        AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits, bitInfo=bit_info)
    except Exception:
        return []
    hits = bit_info.get(int(bit_id), [])
    motifs: List[str] = []
    for atom_idx, rad in hits:
        try:
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, int(rad), int(atom_idx))
            amap: Dict[int, int] = {}
            submol = Chem.PathToSubmol(mol, env, atomMap=amap) if env else Chem.MolFromSmiles(Chem.MolToSmiles(mol))
            if submol is None:
                continue
            smarts = Chem.MolToSmarts(submol)
            if smarts:
                motifs.append(smarts)
        except Exception:
            continue
    return motifs


def _build_fp_demasked_table(
    *,
    imp_df: pd.DataFrame,
    feature_names: List[str],
    feature_types: List[str],
    shap_values: np.ndarray,
    X_use: np.ndarray,
    y_use: np.ndarray,
    ids_use: List[str],
    smiles_use: List[str],
    fp_top_k: int,
    fp_radius: int,
    fp_nbits: int,
) -> pd.DataFrame:
    fp_imp = imp_df[imp_df["feature_type"].astype(str).str.lower() == "fp"].head(int(max(1, fp_top_k)))
    if fp_imp.empty:
        return pd.DataFrame()

    name_to_idx = {str(n): i for i, n in enumerate(feature_names)}
    rows: List[Dict[str, Any]] = []
    n_samples = int(X_use.shape[0]) if X_use is not None else 0

    for _, r in fp_imp.iterrows():
        fname = str(r["feature"])
        bit_id = _parse_fp_bit(fname)
        if bit_id is None or fname not in name_to_idx:
            continue
        col_idx = int(name_to_idx[fname])
        present_mask = np.asarray(X_use[:, col_idx] > 0, dtype=bool)
        present_idx = np.where(present_mask)[0]
        if present_idx.size == 0:
            continue

        shap_col = np.asarray(shap_values[:, col_idx], dtype=float)
        mean_abs_present = float(np.mean(np.abs(shap_col[present_idx])))
        mean_signed_present = float(np.mean(shap_col[present_idx]))
        occurrence = int(present_idx.size)
        occurrence_frac = float(occurrence / n_samples) if n_samples > 0 else float("nan")

        motif_counts: Dict[str, int] = {}
        for i in present_idx.tolist():
            motifs = _bit_motifs_from_smiles(str(smiles_use[i]), bit_id=bit_id, radius=fp_radius, nbits=fp_nbits)
            for m in motifs:
                motif_counts[m] = motif_counts.get(m, 0) + 1
        top_motif = ""
        top_motif_count = 0
        if motif_counts:
            top_motif, top_motif_count = sorted(motif_counts.items(), key=lambda x: x[1], reverse=True)[0]

        active_idx = [i for i in present_idx.tolist() if int(y_use[i]) == 1]
        rep_id = ""
        rep_smiles = ""
        rep_shap = float("nan")
        if active_idx:
            best_i = max(active_idx, key=lambda i: shap_col[i])
            rep_id = str(ids_use[best_i])
            rep_smiles = str(smiles_use[best_i])
            rep_shap = float(shap_col[best_i])

        rows.append(
            {
                "feature": fname,
                "bit_id": int(bit_id),
                "mean_abs_shap_global": float(r["mean_abs_shap"]),
                "mean_abs_shap_when_present": mean_abs_present,
                "mean_signed_shap_when_present": mean_signed_present,
                "occurrence_count": occurrence,
                "occurrence_fraction": occurrence_frac,
                "top_motif_smarts": top_motif,
                "top_motif_count": int(top_motif_count),
                "representative_active_id": rep_id,
                "representative_active_smiles": rep_smiles,
                "representative_active_shap": rep_shap,
            }
        )
    return pd.DataFrame(rows).sort_values("mean_abs_shap_global", ascending=False).reset_index(drop=True)


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

        fp_demasked_df = _build_fp_demasked_table(
            imp_df=imp_df,
            feature_names=feature_names,
            feature_types=feature_types,
            shap_values=shap_values,
            X_use=X_use,
            y_use=y_use,
            ids_use=ids_use,
            smiles_use=smiles_use,
            fp_top_k=int(config.fp_top_k),
            fp_radius=int(config.fp_radius),
            fp_nbits=int(config.fp_nbits),
        )
        fp_demasked_csv = model_out_dir / "fp_motif_demasked.csv"
        if not fp_demasked_df.empty:
            fp_demasked_df.to_csv(fp_demasked_csv, index=False)

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
                "fp_motif_demasked_csv": (str(fp_demasked_csv) if fp_demasked_csv.exists() else ""),
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
    p.add_argument("--fp-top-k", type=int, default=20, help="Top-K fingerprint bits to demask into motifs")
    p.add_argument("--fp-radius", type=int, default=2, help="Morgan radius for bit demasking")
    p.add_argument("--fp-nbits", type=int, default=2048, help="Morgan nBits for bit demasking")
    p.add_argument("--output-dir", type=Path, help="Output directory (default: <split_seed_dir>/shap_analysis)")
    p.add_argument("--plot-only", action="store_true", help="Plot only from existing SHAP exports")
    p.add_argument("--plot-model", help="Model key for plot-only mode (default: first from --models)")
    p.add_argument("--local-id", help="Optional sample id for local explanation in plot-only mode")
    return p.parse_args()


def plot_only_from_exports(
    *,
    run_dir: Path,
    split_seed: int,
    model_key: str,
    output_dir: Optional[Path] = None,
    local_id: Optional[str] = None,
    dpi: int = 600,
) -> None:
    import matplotlib as mpl
    mpl.use("Agg", force=True)
    from matplotlib import pyplot as plt
    import shap  # type: ignore

    out_dir = output_dir or (_resolve_split_dir(run_dir, split_seed) / "shap_analysis")
    model_dir = out_dir / model_key
    npz_path = model_dir / "shap_values_external.npz"
    imp_path = model_dir / "feature_importance.csv"
    fp_demasked_path = model_dir / "fp_motif_demasked.csv"
    if not npz_path.exists() or not imp_path.exists():
        raise FileNotFoundError(f"Missing required SHAP exports under {model_dir}")

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Cambria", "Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "text.color": "black",
        "axes.labelcolor": "black",
        "axes.titlecolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    with np.load(npz_path, allow_pickle=True) as data:
        shap_values = np.asarray(data["shap_values"], dtype=np.float64)
        X = np.asarray(data["X"], dtype=np.float64)
        y_prob = np.asarray(data["y_prob"], dtype=np.float64) if "y_prob" in data.files else np.asarray([])
        ids = [str(x) for x in np.atleast_1d(data["id"]).tolist()]
        base_value = float(np.asarray(data["base_value"], dtype=np.float64).reshape(-1)[0])
        feature_display = [str(x) for x in np.atleast_1d(data["feature_display"]).tolist()]
    X_df = pd.DataFrame(X, columns=feature_display)
    imp_df = pd.read_csv(imp_path)

    def _save(fig, name: str) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(model_dir / f"{name}.png", bbox_inches="tight", dpi=dpi)
        fig.savefig(model_dir / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)

    fig = plt.figure(figsize=(7.2, 5.2))
    shap.summary_plot(shap_values, X_df, feature_names=feature_display, max_display=20, show=False, cmap="viridis")
    plt.xlabel("SHAP value (impact on model output)")
    plt.title(f"{model_key} | SHAP Summary", fontsize=11, pad=10)
    plt.tight_layout()
    _save(fig, "A_global_shap_summary_beeswarm")

    desc_df = imp_df[imp_df["feature_type"].astype(str).str.lower() == "descriptor"].sort_values("mean_abs_shap", ascending=False).head(20)
    fp_df_all = imp_df[imp_df["feature_type"].astype(str).str.lower() == "fp"].sort_values("mean_abs_shap", ascending=False).head(20)
    if not desc_df.empty:
        fig, ax = plt.subplots(figsize=(6.8, 5.6))
        y = np.arange(len(desc_df))
        ax.barh(y, desc_df["mean_abs_shap"].to_numpy(), color="#4C72B0", edgecolor="black", linewidth=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(desc_df["feature_display"].astype(str).tolist())
        ax.invert_yaxis()
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"{model_key} | Descriptor Importance Ranking")
        _save(fig, "B_descriptor_importance_ranking")

    # Combined B+C stacked-right chart (top: descriptor, bottom: fingerprint; both positive)
    if not desc_df.empty and not fp_df_all.empty:
        n_each = int(min(12, len(desc_df), len(fp_df_all)))
        desc_top = desc_df.head(n_each).reset_index(drop=True)
        fp_top = fp_df_all.head(n_each).reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(9.4, 8.6))

        gap = 0
        y_desc = np.arange(n_each)
        y_fp = np.arange(n_each) + n_each + gap
        desc_vals = desc_top["mean_abs_shap"].to_numpy(dtype=float)
        fp_vals = fp_top["mean_abs_shap"].to_numpy(dtype=float)

        ax.barh(y_desc, desc_vals, color="#4C72B0", edgecolor="black", linewidth=0.6, label="Descriptor")
        ax.barh(y_fp, fp_vals, color="#55A868", edgecolor="black", linewidth=0.6, label="Fingerprint")

        ax.set_yticks(np.concatenate([y_desc, y_fp]))
        ax.set_yticklabels(
            desc_top["feature_display"].astype(str).tolist() + fp_top["feature_display"].astype(str).tolist()
        )
        ax.invert_yaxis()
        ax.margins(y=0.01)
        ax.set_ylim(y_fp[-1] + 0.45, -0.45)
        max_abs = float(max(np.max(desc_vals), np.max(fp_vals)))
        min_pos = float(min(np.min(desc_vals), np.min(fp_vals)))
        # Dynamic axis compression: keep a small, data-adaptive headroom only.
        right_pad = max(0.0015, 0.06 * max_abs)
        left_pad = max(0.0, min_pos - max(0.001, 0.02 * max_abs))
        ax.set_xlim(left_pad, max_abs + right_pad)
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"{model_key} | Descriptor and Fingerprint Importance Ranking")
        #ax.text(0.02, 0.98, "(B) Descriptor", transform=ax.transAxes, ha="left", va="top", fontsize=10)
        #ax.text(0.02, 0.48, "(C) Fingerprint", transform=ax.transAxes, ha="left", va="top", fontsize=10)
        ax.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="black", facecolor="white", framealpha=1.0)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)

        fig.tight_layout()
        _save(fig, "BC_stacked_descriptor_fp")

    if fp_demasked_path.exists():
        fp_df = pd.read_csv(fp_demasked_path).head(20)
        if not fp_df.empty:
            fig, ax = plt.subplots(figsize=(8.8, 6.2))
            y = np.arange(len(fp_df))
            ax.barh(y, fp_df["mean_abs_shap_global"].to_numpy(), color="#55A868", edgecolor="black", linewidth=0.6)
            ax.set_yticks(y)
            ax.set_yticklabels([f"bit {int(b)} | f={float(fr):.2f}" for b, fr in zip(fp_df["bit_id"], fp_df["occurrence_fraction"])])
            ax.invert_yaxis()
            ax.set_xlabel("Mean |SHAP value|")
            ax.set_title(f"{model_key} | De-masked Fingerprint Motifs")
            _save(fig, "C_demasked_fp_motifs")

    if shap_values.ndim == 2 and len(shap_values) > 0:
        sample_idx = 0
        if local_id is not None and local_id in ids:
            sample_idx = ids.index(local_id)
        elif y_prob.size == len(shap_values):
            sample_idx = int(np.nanargmax(y_prob))
        contrib = shap_values[sample_idx]
        top_idx = np.argsort(np.abs(contrib))[::-1][:10]
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        vals = contrib[top_idx]
        feats = [feature_display[i] for i in top_idx]
        ax.barh(np.arange(len(top_idx)), vals, color=["#C44E52" if v > 0 else "#4C72B0" for v in vals], edgecolor="black", linewidth=0.5)
        ax.set_yticks(np.arange(len(top_idx)))
        ax.set_yticklabels(feats)
        ax.invert_yaxis()
        ax.axvline(0.0, color="black", linewidth=0.9)
        pred_txt = f"{float(y_prob[sample_idx]):.3f}" if y_prob.size == len(shap_values) else "NA"
        ax.set_title(f"{model_key} | Local Contributors | id={ids[sample_idx]} | p={pred_txt}")
        ax.set_xlabel("SHAP value")
        _save(fig, "D_local_top_contributors")

        fig = plt.figure(figsize=(8.2, 5.2))
        explanation = shap.Explanation(values=contrib, base_values=base_value, data=X[sample_idx], feature_names=feature_display)
        shap.plots.waterfall(explanation, max_display=10, show=False)
        # SHAP may create multiple axes/text artists and override styles; enforce final bounds/colors.
        vertical_text_tokens = {"|", "│", "┃", "┆", "┇"}
        for ax_w in fig.axes:
            try:
                left, _ = ax_w.get_xlim()
                ax_w.set_xlim(left, 1.0)
                ax_w.set_autoscalex_on(False)
            except Exception:
                pass
            try:
                ax_w.tick_params(axis="both", colors="black")
                ax_w.xaxis.label.set_color("black")
                ax_w.yaxis.label.set_color("black")
                ax_w.title.set_color("black")
                for lbl in ax_w.get_xticklabels() + ax_w.get_yticklabels():
                    lbl.set_color("black")
            except Exception:
                pass
            try:
                # Force all line artists to a consistent black style.
                for ln in ax_w.lines:
                    ln.set_color("black")
                    ln.set_linewidth(0.8)
            except Exception:
                pass
            try:
                # SHAP may draw the f(x) marker as LineCollection segments.
                for coll in ax_w.collections:
                    if hasattr(coll, "set_color"):
                        coll.set_color("black")
                    if hasattr(coll, "set_edgecolor"):
                        coll.set_edgecolor("black")
                    if hasattr(coll, "set_linewidth"):
                        coll.set_linewidth(0.8)
            except Exception:
                pass
            try:
                ax_w.spines["top"].set_visible(False)
                ax_w.spines["right"].set_visible(False)
            except Exception:
                pass
        for txt in fig.texts:
            try:
                t = str(txt.get_text()).strip()
                if t in vertical_text_tokens:
                    txt.set_visible(False)
                else:
                    txt.set_color("black")
            except Exception:
                pass
        for ax_w in fig.axes:
            for txt in ax_w.texts:
                try:
                    t = str(txt.get_text()).strip()
                    if t in vertical_text_tokens:
                        txt.set_visible(False)
                    else:
                        txt.set_color("black")
                except Exception:
                    pass
        plt.title(f"{model_key} | Local Waterfall | id={ids[sample_idx]}", fontsize=11, pad=10)
        plt.tight_layout()
        _save(fig, "D_local_waterfall")


def main() -> None:
    args = parse_args()
    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.plot_only:
        target_model = (args.plot_model or (model_list[0] if model_list else "SVC")).upper()
        plot_only_from_exports(
            run_dir=args.run_dir,
            split_seed=int(args.split_seed),
            model_key=target_model,
            output_dir=args.output_dir,
            local_id=args.local_id,
            dpi=600,
        )
        print("[OK] Plot-only export complete")
        print(f"  - Model: {target_model}")
        print(f"  - Output dir: {(args.output_dir or (_resolve_split_dir(args.run_dir, int(args.split_seed)) / 'shap_analysis')) / target_model}")
        return

    cfg = ShapConfig(
        run_dir=args.run_dir,
        split_seed=int(args.split_seed),
        models=model_list,
        task=str(args.task),
        max_samples=(int(args.max_samples) if args.max_samples else None),
        random_state=int(args.random_state),
        feature_name_mode=str(args.feature_name_mode),
        kernel_background_samples=int(args.kernel_background_samples),
        fp_top_k=int(args.fp_top_k),
        fp_radius=int(args.fp_radius),
        fp_nbits=int(args.fp_nbits),
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
        "font_family": "Cambria", # Cambria, Times New Roman
        "font_size": 10,
        "dpi": 600,
        "max_display": 20,
        "heatmap_samples": 64,
        "local_top_k": 10,
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
    OUT_DIR = Path("../models_out/qsar_ml_20260412_162829/split_seed_12345/shap_analysis")     # Relative Path
    MODEL_KEY = "SVC"   # "LR" or "SVC"

    npz_path = OUT_DIR / MODEL_KEY / "shap_values_external.npz"
    imp_path = OUT_DIR / MODEL_KEY / "feature_importance.csv"
    fp_demasked_path = OUT_DIR / MODEL_KEY / "fp_motif_demasked.csv"

    if not npz_path.exists():
        raise FileNotFoundError(f"Missing SHAP exports: {npz_path}")
    if not imp_path.exists():
        raise FileNotFoundError(f"Missing feature importance: {imp_path}")

    if shap is None:
        raise ImportError("Install SHAP: pip install shap")

    # Load data
    with np.load(npz_path, allow_pickle=True) as data:
        shap_values = np.asarray(data["shap_values"], dtype=np.float64)
        X = np.asarray(data["X"], dtype=np.float64)
        y_prob = np.asarray(data["y_prob"], dtype=np.float64) if "y_prob" in data.files else np.asarray([])
        ids = [str(x) for x in np.atleast_1d(data["id"]).tolist()]
        smiles = [str(x) for x in np.atleast_1d(data["smiles"]).tolist()]
        y_true = np.asarray(data["y_true"], dtype=np.float64) if "y_true" in data.files else np.asarray([])
        base_value = float(np.asarray(data["base_value"], dtype=np.float64).reshape(-1)[0])
        feature_names_raw = [str(x) for x in np.atleast_1d(data["feature_names"]).tolist()]
        feature_types = [str(x) for x in np.atleast_1d(data["feature_types"]).tolist()]
        feature_display = [str(x) for x in np.atleast_1d(data["feature_display"]).tolist()]

    X_df = pd.DataFrame(X, columns=feature_display)
    imp_df = pd.read_csv(imp_path)


    def _save_fig(fig, name: str):
        out_dir = OUT_DIR / MODEL_KEY
        out_dir.mkdir(parents=True, exist_ok=True)

        fig.savefig(out_dir / f"{name}.png", bbox_inches="tight", dpi=PLOT_STYLE["dpi"])
        fig.savefig(out_dir / f"{name}.svg", bbox_inches="tight")

    # (A) Global SHAP summary plot (top-20; descriptor+fp mixed; colored by feature value)
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
    _save_fig(fig, "A_global_shap_summary_beeswarm")
    plt.show()

    # (B) Descriptor importance ranking (mean |SHAP|)
    desc_df = (
        imp_df[imp_df["feature_type"].astype(str).str.lower() == "descriptor"]
        .sort_values("mean_abs_shap", ascending=False)
        .head(20)
    )
    if not desc_df.empty:
        fig, ax = plt.subplots(figsize=(6.8, 5.6))
        y = np.arange(len(desc_df))
        ax.barh(y, desc_df["mean_abs_shap"].to_numpy(), color="#4C72B0", edgecolor="black", linewidth=0.6, alpha=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels(desc_df["feature_display"].astype(str).tolist())
        ax.invert_yaxis()
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"{MODEL_KEY} | Descriptor Importance Ranking")
        ax.grid(False)
        plt.tight_layout()
        _save_fig(fig, "B_descriptor_importance_ranking")
        plt.show()

    # (C) De-masked fingerprint motifs summary
    if fp_demasked_path.exists():
        fp_df = pd.read_csv(fp_demasked_path).head(20)
        if not fp_df.empty:
            fig, ax = plt.subplots(figsize=(8.8, 6.2))
            y = np.arange(len(fp_df))
            ax.barh(y, fp_df["mean_abs_shap_global"].to_numpy(), color="#55A868", edgecolor="black", linewidth=0.6, alpha=0.9)
            labels = []
            for _, row in fp_df.iterrows():
                labels.append(f"bit {int(row['bit_id'])} | f={float(row['occurrence_fraction']):.2f}")
            ax.set_yticks(y)
            ax.set_yticklabels(labels)
            ax.invert_yaxis()
            ax.set_xlabel("Mean |SHAP value|")
            ax.set_title(f"{MODEL_KEY} | De-masked Fingerprint Motifs")
            for yi, (_, row) in enumerate(fp_df.iterrows()):
                txt = f"SMARTS: {str(row.get('top_motif_smarts', ''))}\nRep active: {str(row.get('representative_active_id', 'NA'))}"
                ax.text(float(row["mean_abs_shap_global"]) + 0.001, yi, txt, va="center", fontsize=7)
            ax.grid(False)
            plt.tight_layout()
            _save_fig(fig, "C_demasked_fp_motifs")
            plt.show()
    else:
        print(f"[WARN] Missing fp motif demasked csv: {fp_demasked_path}")

    # (D) Local explanation example (single compound)
    if shap_values.ndim == 2 and len(shap_values) > 0:
        if y_prob.size == len(shap_values):
            sample_idx = int(np.nanargmax(y_prob))
        else:
            sample_idx = 0

        contrib = shap_values[sample_idx]
        top_k = int(PLOT_STYLE["local_top_k"])
        top_idx = np.argsort(np.abs(contrib))[::-1][:top_k]
        top_features = [feature_display[i] for i in top_idx]
        top_values = contrib[top_idx]

        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        colors = ["#C44E52" if v > 0 else "#4C72B0" for v in top_values]
        ypos = np.arange(len(top_idx))
        ax.barh(ypos, top_values, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_yticks(ypos)
        ax.set_yticklabels(top_features)
        ax.invert_yaxis()
        ax.axvline(0.0, color="black", linewidth=0.9)
        pred_txt = f"{float(y_prob[sample_idx]):.3f}" if y_prob.size == len(shap_values) else "NA"
        title = f"{MODEL_KEY} | Local Contributors | id={ids[sample_idx]} | p={pred_txt}"
        ax.set_title(title)
        ax.set_xlabel("SHAP value")
        ax.grid(False)
        plt.tight_layout()
        _save_fig(fig, "D_local_top_contributors")
        plt.show()

        try:
            explanation = shap.Explanation(
                values=contrib,
                base_values=base_value,
                data=X[sample_idx],
                feature_names=feature_display,
            )
            fig = plt.figure(figsize=(8.2, 5.2))
            shap.plots.waterfall(explanation, max_display=top_k, show=False)
            plt.title(f"{MODEL_KEY} | Local Waterfall | id={ids[sample_idx]}", fontsize=11, pad=10)
            plt.tight_layout()
            _save_fig(fig, "D_local_waterfall")
            plt.show()
        except Exception as e:
            print(f"[WARN] Local waterfall skipped: {e}")
# %%
