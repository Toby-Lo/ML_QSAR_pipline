"""Calibrate QSAR probability outputs from step10_qsar_ml.py artifacts.

Usage:
  python scripts/step20_calibration.py \
    --run-dir models_out/qsar_ml_20260409_120000 \
    --input data/test_data_complex_with_fingerprints.csv \
    --methods both \
    --calibration-source dev

This will read the trained models from the specified run directory
split_seed_*/models/full_dev/{MODEL}/seed_{seed}/model.joblib
split_seed_*/feature_processors/fp_mask.npy
split_seed_*/feature_processors/descriptor_names.json
split_seed_*/split_indices.json
split_seed_*/predictions/cv_predictions_fold_*.csv

options: 
    method="sigmoid"; "isotonic"; or "both" 
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

try:
    from rdkit.Chem.rdMolDescriptors import MorganGenerator

    _MORGAN_GENERATOR_AVAILABLE = True
except ImportError:
    from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect

    _MORGAN_GENERATOR_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate trained QSAR models from step10 outputs")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory from step10 (contains split_seed_*)")
    parser.add_argument("--input", type=Path, required=True, help="Input CSV/Parquet used by step10")
    parser.add_argument("--methods", choices=["sigmoid", "isotonic", "both"], default="both")
    parser.add_argument("--calibration-source", choices=["dev", "external"], default="dev")
    parser.add_argument("--cv-folds", type=int, default=5, help="Grouped CV folds for calibration")
    parser.add_argument("--split-seeds", help="Comma-separated seeds; default auto-detect from run-dir")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--bins", type=int, default=10, help="Number of bins for reliability curve")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def setup_logger(run_dir: Path) -> logging.Logger:
    log_dir = run_dir / "logs"
    ensure_dir(log_dir)
    logger = logging.getLogger("step20_calibration")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_dir / "step20_calibration.log")
    sh = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


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


def build_grouped_splits(y: np.ndarray, groups: np.ndarray, n_splits: int, random_state: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    n_splits = max(2, int(n_splits))
    unique_groups = np.unique(groups)
    n_splits = min(n_splits, len(unique_groups))
    if n_splits < 2:
        raise ValueError("Not enough unique scaffolds to build grouped CV splits")
    try:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(sgkf.split(X=np.zeros(len(y)), y=y, groups=groups))
        if any(np.unique(y[val_idx]).shape[0] < 2 for _, val_idx in splits):
            raise ValueError("single-class validation fold with StratifiedGroupKFold")
        return splits
    except Exception:
        gkf = GroupKFold(n_splits=n_splits)
        return list(gkf.split(X=np.zeros(len(y)), y=y, groups=groups))


def load_cv_prediction_brier(split_seed_dir: Path) -> Dict[str, float]:
    pred_dir = split_seed_dir / "predictions"
    rows: Dict[str, List[float]] = {}
    for path in sorted(pred_dir.glob("cv_predictions_fold_*.csv")):
        df = pd.read_csv(path)
        if not {"model", "y_true", "y_prob"}.issubset(df.columns):
            continue
        for model, grp in df.groupby("model"):
            valid = grp["y_prob"].notna()
            if valid.sum() == 0:
                continue
            score = brier_score_loss(grp.loc[valid, "y_true"].astype(int), grp.loc[valid, "y_prob"].astype(float))
            rows.setdefault(str(model), []).append(float(score))
    return {model: float(np.mean(vals)) for model, vals in rows.items() if vals}


def reliability_plot(y_true: np.ndarray,
                     y_prob_raw: np.ndarray,
                     y_prob_cal: np.ndarray,
                     title: str,
                     out_path: Path,
                     n_bins: int = 10) -> None:
    frac_raw, mean_raw = calibration_curve(y_true, y_prob_raw, n_bins=n_bins, strategy="quantile")
    frac_cal, mean_cal = calibration_curve(y_true, y_prob_cal, n_bins=n_bins, strategy="quantile")
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfectly calibrated")
    plt.plot(mean_raw, frac_raw, "o-", label="Raw")
    plt.plot(mean_cal, frac_cal, "o-", label="Calibrated")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def calibrate_one_model(model,
                        X_cal: np.ndarray,
                        y_cal: np.ndarray,
                        groups_cal: np.ndarray,
                        method: str,
                        cv_folds: int,
                        random_state: int):
    cv_splits = build_grouped_splits(y_cal, groups_cal, n_splits=cv_folds, random_state=random_state)
    calibrated = CalibratedClassifierCV(estimator=clone(model), method=method, cv=cv_splits)
    calibrated.fit(X_cal, y_cal)
    return calibrated


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    logger = setup_logger(run_dir)
    methods = ["sigmoid", "isotonic"] if args.methods == "both" else [args.methods]

    if args.split_seeds:
        seeds = [int(x) for x in args.split_seeds.split(",") if x.strip()]
        split_dirs = [run_dir / f"split_seed_{seed}" for seed in seeds]
    else:
        split_dirs = sorted([p for p in run_dir.glob("split_seed_*") if p.is_dir()])
    split_dirs = [p for p in split_dirs if p.exists()]
    if not split_dirs:
        raise SystemExit("No split_seed_* directories found")

    logger.info(f"Loading input data from {args.input}")
    df = read_table(args.input)
    if args.smiles_column not in df.columns or args.label_column not in df.columns:
        raise ValueError("Input data missing required smiles/label columns")
    smiles_all = df[args.smiles_column].fillna("").astype(str).tolist()
    y_all = pd.to_numeric(df[args.label_column], errors="coerce").to_numpy(dtype=int)

    # Descriptor schema is expected to be consistent across split seeds.
    descriptor_file = split_dirs[0] / "feature_processors" / "descriptor_names.json"
    descriptor_names = json.loads(descriptor_file.read_text())

    fp_matrix = detect_existing_fingerprints(df)
    if fp_matrix is None:
        logger.info("No precomputed fingerprints found in input; computing Morgan fingerprints.")
        fp_matrix = compute_morgan_fingerprints(smiles_all)
    if set(descriptor_names).issubset(df.columns):
        desc_matrix = df[descriptor_names].astype(np.float32).fillna(0.0).to_numpy(dtype=np.float32)
    else:
        logger.info("Descriptor columns missing in input; recomputing RDKit descriptors.")
        desc_matrix = compute_rdkit_descriptors(smiles_all, descriptor_names)

    summary_rows: List[Dict[str, Any]] = []
    cv_brier_rows: List[Dict[str, Any]] = []

    for split_dir in split_dirs:
        split_seed = int(split_dir.name.split("_")[-1])
        logger.info(f"Processing {split_dir.name}")
        split_idx = json.loads((split_dir / "split_indices.json").read_text())
        cal_indices = split_idx["train"] if args.calibration_source == "dev" else split_idx["external"]
        cal_indices = np.array(cal_indices, dtype=int)
        y_cal = y_all[cal_indices]
        smiles_cal = [smiles_all[i] for i in cal_indices]
        groups_cal = np.array([get_scaffold(s) for s in smiles_cal], dtype=object)

        mask = np.load(split_dir / "feature_processors" / "fp_mask.npy")
        fp_cal = fp_matrix[cal_indices][:, mask]
        desc_cal_raw = desc_matrix[cal_indices]

        cv_brier = load_cv_prediction_brier(split_dir)
        for model_key, score in cv_brier.items():
            cv_brier_rows.append({
                "split_seed": split_seed,
                "model": model_key,
                "cv_prediction_brier_mean": score,
            })

        model_roots = sorted([p for p in (split_dir / "models" / "full_dev").glob("*") if p.is_dir()])
        for model_root in model_roots:
            model_key = model_root.name
            model_dir = model_root / f"seed_{split_seed}"
            if not model_dir.exists():
                continue
            model = joblib.load(model_dir / "model.joblib")
            scaler_path = model_dir / "scaler.joblib"
            if scaler_path.exists():
                scaler = joblib.load(scaler_path)
                desc_cal = scaler.transform(desc_cal_raw.astype(np.float32))
            else:
                desc_cal = desc_cal_raw.astype(np.float32)
            X_cal = np.concatenate([fp_cal, desc_cal], axis=1).astype(np.float32)

            if not hasattr(model, "predict_proba"):
                logger.warning(f"Skipping {model_key} in {split_dir.name}: no predict_proba")
                continue
            raw_prob = model.predict_proba(X_cal)[:, 1]
            raw_brier = float(brier_score_loss(y_cal, raw_prob))

            for method in methods:
                calib_out_dir = split_dir / "calibration" / model_key / f"method_{method}"
                ensure_dir(calib_out_dir)
                try:
                    calibrated = calibrate_one_model(
                        model=model,
                        X_cal=X_cal,
                        y_cal=y_cal,
                        groups_cal=groups_cal,
                        method=method,
                        cv_folds=args.cv_folds,
                        random_state=args.random_state + split_seed,
                    )
                except Exception as exc:
                    logger.warning(f"Calibration failed for {split_dir.name}/{model_key}/{method}: {exc}")
                    continue

                cal_prob = calibrated.predict_proba(X_cal)[:, 1]
                cal_brier = float(brier_score_loss(y_cal, cal_prob))
                frac_raw, mean_raw = calibration_curve(y_cal, raw_prob, n_bins=args.bins, strategy="quantile")
                frac_cal, mean_cal = calibration_curve(y_cal, cal_prob, n_bins=args.bins, strategy="quantile")

                reliability_plot(
                    y_true=y_cal,
                    y_prob_raw=raw_prob,
                    y_prob_cal=cal_prob,
                    title=f"{split_dir.name} | {model_key} | {method}",
                    out_path=calib_out_dir / "reliability_plot.png",
                    n_bins=args.bins,
                )

                curve_df = pd.DataFrame({
                    "mean_pred_raw": mean_raw,
                    "frac_pos_raw": frac_raw,
                })
                if len(mean_cal) == len(curve_df):
                    curve_df["mean_pred_cal"] = mean_cal
                    curve_df["frac_pos_cal"] = frac_cal
                else:
                    curve_df = pd.DataFrame({
                        "mean_pred_raw": mean_raw,
                        "frac_pos_raw": frac_raw,
                        "mean_pred_cal": pd.Series(mean_cal),
                        "frac_pos_cal": pd.Series(frac_cal),
                    })
                curve_df.to_csv(calib_out_dir / "calibration_curve.csv", index=False)

                metrics = {
                    "split_seed": split_seed,
                    "model": model_key,
                    "method": method,
                    "calibration_source": args.calibration_source,
                    "n_samples": int(len(y_cal)),
                    "brier_raw": raw_brier,
                    "brier_calibrated": cal_brier,
                    "brier_improvement": raw_brier - cal_brier,
                }
                (calib_out_dir / "calibration_metrics.json").write_text(json.dumps(metrics, indent=2))
                joblib.dump(calibrated, calib_out_dir / "calibrated_model.joblib")
                summary_rows.append(metrics)
                logger.info(
                    f"{split_dir.name} {model_key} {method}: "
                    f"Brier raw={raw_brier:.4f}, cal={cal_brier:.4f}"
                )

    results_dir = run_dir / "results"
    ensure_dir(results_dir)
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(results_dir / "calibration_summary.csv", index=False)
        logger.info(f"Saved calibration summary: {results_dir / 'calibration_summary.csv'}")
    if cv_brier_rows:
        pd.DataFrame(cv_brier_rows).drop_duplicates().to_csv(
            results_dir / "calibration_cv_prediction_brier.csv", index=False
        )
        logger.info(f"Saved CV prediction Brier summary: {results_dir / 'calibration_cv_prediction_brier.csv'}")


if __name__ == "__main__":
    main()
