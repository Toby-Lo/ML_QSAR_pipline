#!/usr/bin/env python3
"""
Production-grade QSAR virtual screening inference (artifact-aligned).

This script performs large-scale inference over a precomputed feature table
(`zinc_features.parquet`) using artifacts produced by `scripts/step10_qsar_ml.py`.

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

I/O + performance
- Streaming read with `pyarrow.parquet.ParquetFile.iter_batches`
- Streaming write with `pyarrow.parquet.ParquetWriter` (zstd)
- tqdm progress bar over batches

python scripts/step33_vs_inference.py \
  --model_dir ./models_out/qsar_ml_20260410_124055 \
  --model_name SVC \
  --seed 12345 \
  --calibration isotonic \
  --threshold auto \
  --threshold_metric youden \
  --input ./data/database/zinc_features.parquet

  # threshold optional: f1(default currently), youden, mcc, recall, precision, or specific value
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union


def _require_deps() -> None:
    try:
        import numpy as _np  # noqa: F401
        import pandas as _pd  # noqa: F401
        import pyarrow as _pa  # noqa: F401
        import pyarrow.parquet as _pq  # noqa: F401
        import tqdm as _tqdm  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Missing runtime dependencies. Please install: numpy, pandas, pyarrow, tqdm, joblib.\n"
            f"Import error: {exc}"
        ) from exc


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
    import pandas as pd

    s = smiles_col
    if not isinstance(s, pd.Series):
        s = pd.Series(s)
    # Keep as python strings; empty/None are invalid.
    s = s.astype("string")
    return s


def select_required_input_columns(parquet_schema_names: Sequence[str], plan: FeaturePlan) -> List[str]:
    names_set = set(parquet_schema_names)
    required_cols: Set[str] = {"zinc_id", "smiles"}

    # Fingerprints: map fp_<idx> -> morgan_<idx> in zinc_features.parquet.
    for col in plan.fp_input_columns:
        if col not in names_set:
            raise KeyError(f"Missing required fingerprint column in input parquet: {col}")
        required_cols.add(col)

    # Descriptors (must exist as-is; no derived features allowed).
    for feat in plan.descriptor_names:
        if feat not in names_set:
            raise KeyError(f"Missing required descriptor column in input parquet: {feat}")
        required_cols.add(feat)

    return [c for c in parquet_schema_names if c in required_cols]


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


def predict_batch(
    df: "pandas.DataFrame",
    plan: FeaturePlan,
    model: Any,
    scaler: Any,
    model_name: str,
    threshold: float,
    smiles_validation: str,
) -> Tuple["pandas.DataFrame", Dict[str, int]]:
    import numpy as np
    import pandas as pd

    n_in = int(len(df))
    if n_in == 0:
        return pd.DataFrame(columns=["zinc_id", "smiles", "prob", "pred_label"]), {
            "processed": 0,
            "predicted": 0,
            "skipped": 0,
            "skipped_nan": 0,
            "skipped_smiles": 0,
        }

    zinc_id = pd.to_numeric(df["zinc_id"], errors="coerce")
    smiles = _normalize_smiles_series(df["smiles"])
    valid = zinc_id.notna() & smiles.notna() & (smiles.str.len() > 0)
    skipped_base = int((~valid).sum())
    if valid.sum() == 0:
        return pd.DataFrame(columns=["zinc_id", "smiles", "prob", "pred_label"]), {
            "processed": n_in,
            "predicted": 0,
            "skipped": skipped_base,
            "skipped_nan": 0,
            "skipped_smiles": 0,
        }

    dfv = df.loc[valid]
    zinc_id_v = zinc_id.loc[valid].astype("int64")
    smiles_v = smiles.loc[valid].astype("string")

    fp_block, desc_block = build_feature_matrices(dfv, plan)
    nan_rows = np.isnan(desc_block).any(axis=1)
    n_skipped_nan = int(nan_rows.sum())

    ok_smiles = np.ones((len(dfv),), dtype=bool)
    if smiles_validation == "rdkit":
        try:
            from rdkit import Chem  # type: ignore
        except Exception:
            Chem = None  # type: ignore
        if Chem is not None:
            # Note: this is a per-row check; enable only when needed.
            smi_list = smiles_v.to_numpy(dtype="object", copy=False).tolist()
            ok_smiles = np.fromiter((Chem.MolFromSmiles(str(s)) is not None for s in smi_list), dtype=bool, count=len(smi_list))
        else:
            ok_smiles = np.ones((len(dfv),), dtype=bool)

    final_ok = (~nan_rows) & ok_smiles
    n_skipped_smiles = int((~ok_smiles & ~nan_rows).sum())
    if final_ok.sum() == 0:
        skipped_total = skipped_base + int((~final_ok).sum())
        return pd.DataFrame(columns=["zinc_id", "smiles", "prob", "pred_label"]), {
            "processed": n_in,
            "predicted": 0,
            "skipped": skipped_total,
            "skipped_nan": n_skipped_nan,
            "skipped_smiles": n_skipped_smiles,
        }

    fp_ok = fp_block[final_ok]
    desc_ok = desc_block[final_ok]
    zinc_id_ok = zinc_id_v.to_numpy(dtype="int64", copy=False)[final_ok]
    smiles_ok = smiles_v.to_numpy(dtype="object", copy=False)[final_ok]

    X = apply_scaling_if_needed(fp_ok, desc_ok, scaler=scaler, model_name=model_name)

    try:
        proba = model.predict_proba(X)[:, 1]
    except Exception as exc:
        raise RuntimeError(f"Model inference failed for batch (n={len(fp_ok)}): {exc}") from exc

    proba = np.asarray(proba, dtype=np.float32)
    proba = np.clip(proba, 0.0, 1.0, out=proba)
    pred = (proba >= float(threshold)).astype(np.int8, copy=False)

    out = pd.DataFrame(
        {
            "zinc_id": zinc_id_ok,
            "smiles": smiles_ok,
            "prob": proba.astype(np.float32, copy=False),
            "pred_label": pred.astype(np.int8, copy=False),
        }
    )
    return out, {
        "processed": n_in,
        "predicted": int(len(out)),
        "skipped": skipped_base + int((~final_ok).sum()),
        "skipped_nan": n_skipped_nan,
        "skipped_smiles": n_skipped_smiles,
    }


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
) -> None:
    import logging
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tqdm import tqdm
    
    logger = logging.getLogger(__name__)

    pf = pq.ParquetFile(input_path)
    cols = select_required_input_columns(pf.schema.names, plan)

    total_rows = int(getattr(pf.metadata, "num_rows", 0) or 0)
    total_batches = (total_rows + batch_size - 1) // batch_size if total_rows > 0 else None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    out_schema = pa.schema(
        [
            pa.field("zinc_id", pa.int64()),
            pa.field("smiles", pa.string()),
            pa.field("prob", pa.float32()),
            pa.field("pred_label", pa.int8()),
            pa.field("model_name", pa.string()),
            pa.field("seed", pa.int64()),
            pa.field("threshold_used", pa.float32()),
        ]
    )

    writer = pq.ParquetWriter(output_path, out_schema, compression="zstd")
    processed = predicted = skipped = 0
    skipped_nan = skipped_smiles = 0

    pbar = tqdm(
        pf.iter_batches(batch_size=int(batch_size), columns=cols, use_threads=True),
        total=total_batches,
        desc="QSAR Inference",
        unit="batch",
    )
    for batch_idx, batch in enumerate(pbar):
        try:
            df = batch.to_pandas()
        except Exception as exc:
            logger.error(f"[Batch {batch_idx}] ERROR converting to pandas: {exc} -> skipped batch")
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
            )
        except Exception as exc:
            logger.error(f"[Batch {batch_idx}] ERROR during inference: {exc} -> skipped batch")
            continue

        processed += int(stats["processed"])
        predicted += int(stats["predicted"])
        skipped += int(stats["skipped"])
        skipped_nan += int(stats.get("skipped_nan", 0))
        skipped_smiles += int(stats.get("skipped_smiles", 0))

        if len(out_df) == 0:
            if batch_idx % 10 == 0:
                logger.info(
                    f"[Batch {batch_idx}] processed={stats['processed']:,} predicted=0 skipped={stats['skipped']:,} "
                    f"(nan={stats.get('skipped_nan',0):,} smiles={stats.get('skipped_smiles',0):,})"
                )
            continue

        # Add constant metadata columns without per-row loops
        out_df["model_name"] = str(model_name)
        out_df["seed"] = int(seed)
        out_df["threshold_used"] = np.float32(threshold)

        table = pa.Table.from_pandas(out_df, schema=out_schema, preserve_index=False)
        writer.write_table(table)

        if batch_idx % 10 == 0:
            logger.info(
                f"[Batch {batch_idx}] processed={stats['processed']:,} predicted={stats['predicted']:,} skipped={stats['skipped']:,} "
                f"(cum_predicted={predicted:,})"
            )

    writer.close()
    logger.info(
        "Inference complete.\n"
        f"  Input : {input_path}\n"
        f"  Output: {output_path}\n"
        f"  Rows  : processed={processed:,} predicted={predicted:,} skipped={skipped:,} "
        f"(nan={skipped_nan:,} smiles={skipped_smiles:,})\n"
        f"  Model : {model_name} | seed={seed} | threshold={threshold:.6f}"
    )


def sanity_check_first_batch(
    pf,
    cols: List[str],
    plan: FeaturePlan,
    expected_dim: int,
    sample_rows: int = 1000,
) -> None:
    import logging
    import numpy as np

    logger = logging.getLogger(__name__)
    it = pf.iter_batches(batch_size=int(sample_rows), columns=cols, use_threads=True)
    first = next(it, None)
    if first is None:
        raise RuntimeError("Input parquet appears empty; cannot run sanity check.")
    df = first.to_pandas()
    fp_u8, desc_f32 = build_feature_matrices(df, plan)
    logger.info("Sanity check:")
    logger.info(f"  FP shape   : {fp_u8.shape} dtype={fp_u8.dtype}")
    logger.info(f"  DESC shape : {desc_f32.shape} dtype={desc_f32.dtype}")
    logger.info(f"  NaN count  : {int(np.isnan(desc_f32).sum())}")
    if fp_u8.shape[1] != plan.n_fp:
        raise RuntimeError(f"Fingerprint dim mismatch: got {fp_u8.shape[1]}, expected {plan.n_fp}")
    if desc_f32.shape[1] != plan.n_desc:
        raise RuntimeError(f"Descriptor dim mismatch: got {desc_f32.shape[1]}, expected {plan.n_desc}")
    if plan.n_features_total != int(expected_dim):
        raise RuntimeError(f"Training feature dim mismatch: plan {plan.n_features_total} vs expected {expected_dim}")
    logger.info(f"  total_dim  : {plan.n_features_total} (matches training)")


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
    )


if __name__ == "__main__":
    # Avoid tokenizers/BLAS oversubscription on big batches
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()
