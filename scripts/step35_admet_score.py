#!/usr/bin/env python3
"""
Step35: ADMET scoring layer (NO hard filtering).

This script merges ADMETlab 3.0 batch predictions onto QSAR/AD results and produces:
- normalized endpoint scores in [0, 1]
- sub-scores (absorption/distribution/metabolism/toxicity)
- admet_score in [0, 1]
- final_score_raw and final_score (min-max normalized to [0, 1])

It is designed as a production scoring layer that integrates into ranking.

python ./scripts/step35_admet_score.py  \
    --input-parquet ./models_out/qsar_ml_20260412_162829/virtual_screening/ad_screening_results_20260419_140044/top1000_full_info.parquet \
    --admetlab-file ./models_out/qsar_ml_20260412_162829/virtual_screening/admet/qsar_top1000_admet.csv \
    --admet-smiles-col raw_smiles \
    --mode default \
    --no-include-bbb \
    --include-dili

[A1A0M]
python ./scripts/step35_admet_score.py  \
    --input-parquet ./models_out/qsar_ml_20260412_162829/virtual_screening/A1A0M_inference_admet/A1A0M_inference_result.parquet \
    --admetlab-file ./models_out/qsar_ml_20260412_162829/virtual_screening/A1A0M_inference_admet/ADMETlab3_result.csv \
    --admet-smiles-col smiles \
    --mode default \
    --no-include-bbb \
    --include-dili

--mode: option["default", "strict", "lenient"]
--no-include-dili: Do not include DILI endpoints. (default: False, --include-dili to include))
--no-include-bbb: Do not include BBB endpoints. (default: False, --include-bbb to include))
--make-plots: Make plots. (default: False)
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EndpointSpec:
    key: str
    aliases: Tuple[str, ...]
    direction: str  # "higher_better" | "lower_better"
    kind: str       # "prob" | "percent" | "continuous"


ENDPOINTS: Tuple[EndpointSpec, ...] = (
    # Absorption
    # ADMETlab 3.0 columns (as in your qsar_top1000_admet.csv) commonly use lowercase names like `hia`, `caco2`.
    EndpointSpec("hia", ("hia", "HIA", "HIA+", "Human Intestinal Absorption", "HIA_prob", "HIA (prob)"), "higher_better", "prob"),
    EndpointSpec("caco2", ("caco2", "Caco2", "Caco-2", "Caco-2 permeability", "Caco2_permeability", "Caco2 (logPapp)"), "higher_better", "continuous"),
    # Solubility (ADMETlab column: logS). Higher logS => better solubility.
    EndpointSpec("logs", ("logS", "logs", "Solubility", "Solubility (logS)"), "higher_better", "continuous"),
    # Distribution
    EndpointSpec("ppb", ("PPB", "Plasma Protein Binding", "PPB(%)", "PPB_percent"), "higher_better", "percent"),
    EndpointSpec("bbb", ("BBB", "Blood Brain Barrier", "BBB_prob", "BBB (prob)"), "higher_better", "prob"),
    # Excretion (optional)
    EndpointSpec("cl_plasma", ("cl-plasma", "cl_plasma", "CL_plasma", "Clearance plasma", "cl (plasma)"), "lower_better", "continuous"),
    EndpointSpec("t_half", ("t0.5", "t_half", "t1/2", "half_life", "t0.5 (h)"), "higher_better", "continuous"),
    # Metabolism
    # ADMETlab 3.0 uses hyphenated endpoint names: `CYP3A4-inh`, `CYP2D6-inh`.
    EndpointSpec(
        "cyp3a4_inh",
        ("CYP3A4-inh", "CYP3A4 inhibitor", "CYP3A4 inhibition", "CYP3A4_inhibition", "CYP3A4_inh", "CYP3A4 (inh prob)"),
        "lower_better",
        "prob",
    ),
    EndpointSpec(
        "cyp2d6_inh",
        ("CYP2D6-inh", "CYP2D6 inhibitor", "CYP2D6 inhibition", "CYP2D6_inhibition", "CYP2D6_inh", "CYP2D6 (inh prob)"),
        "lower_better",
        "prob",
    ),
    # Toxicity
    EndpointSpec("herg", ("hERG", "hERG blocker", "hERG_blocker", "hERG_inhibition", "hERG (risk)"), "lower_better", "prob"),
    EndpointSpec("ames", ("AMES", "Ames", "AMES mutagenicity", "AMES (risk)", "Ames (risk)"), "lower_better", "prob"),
    EndpointSpec("dili", ("DILI", "Drug-induced liver injury", "DILI (risk)", "DILI_risk"), "lower_better", "prob"),
)

# =========================
# Tunable scoring config
# =========================
#
# Component weights control the final ADMET score mixture.
# By default we keep the original prompt weights (Abs/Dist/Met/Tox) and give Excretion weight 0.
DEFAULT_COMPONENT_WEIGHTS: Dict[str, float] = {
    "absorption": 0.25,
    "distribution": 0.15,
    "metabolism": 0.25,
    "excretion": 0.0,  # optional, default off
    "toxicity": 0.35,
}

# Within-component endpoint weights (should sum to ~1.0 within each component).
DEFAULT_ENDPOINT_WEIGHTS: Dict[str, Dict[str, float]] = {
    # absorption = 0.4*caco2 + 0.3*hia + 0.3*logS
    "absorption": {"caco2": 0.4, "hia": 0.3, "logs": 0.3},
    "distribution": {"ppb": 0.8, "bbb": 0.2},  # BBB is optional and intentionally low-weight.
    "metabolism": {"cyp3a4_inh": 0.5, "cyp2d6_inh": 0.5},
    "excretion": {"cl_plasma": 0.5, "t_half": 0.5},
    "toxicity": {"herg": 1 / 3, "ames": 1 / 3, "dili": 1 / 3},
}


def _mode_component_weights(mode: str) -> Dict[str, float]:
    """
    Preset component weights by mode.
    - default: matches the user's prompt (Abs 0.25 / Dist 0.15 / Met 0.25 / Tox 0.35)
    - strict: toxicity heavier
    - lenient: toxicity lighter, absorption/distribution heavier
    Excretion defaults to 0 unless you override it in the weight config.
    """
    mode = str(mode).strip().lower()
    if mode == "strict":
        return {"absorption": 0.20, "distribution": 0.10, "metabolism": 0.25, "excretion": 0.0, "toxicity": 0.45}
    if mode == "lenient":
        return {"absorption": 0.30, "distribution": 0.20, "metabolism": 0.25, "excretion": 0.0, "toxicity": 0.25}
    return dict(DEFAULT_COMPONENT_WEIGHTS)


def _load_weight_config(
    path: Optional[Path],
    mode: str,
    logger: logging.Logger,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    Optional JSON override for weights.

    Expected schema:
      {
        "component_weights": {"absorption": 0.25, "distribution": 0.15, "metabolism": 0.25, "excretion": 0.0, "toxicity": 0.35},
        "endpoint_weights": {
          "absorption": {"hia": 0.5, "caco2": 0.5},
          "distribution": {"ppb": 0.8, "bbb": 0.2},
          ...
        }
      }
    """
    comp = _mode_component_weights(mode)
    endp = {k: dict(v) for k, v in DEFAULT_ENDPOINT_WEIGHTS.items()}
    if path is None:
        return comp, endp
    if not path.exists():
        raise FileNotFoundError(f"Weight config not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if isinstance(data.get("component_weights"), dict):
            for k, v in data["component_weights"].items():
                if k in comp:
                    comp[k] = float(v)
        if isinstance(data.get("endpoint_weights"), dict):
            for comp_k, weights in data["endpoint_weights"].items():
                if comp_k in endp and isinstance(weights, dict):
                    for ek, ev in weights.items():
                        endp[comp_k][ek] = float(ev)
    logger.info(f"Loaded weight config: {path}")
    return comp, endp


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("step35_admet")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(h)
    return logger


def _resolve_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    # Case-insensitive fallback (common in exported spreadsheets)
    lowered = {c.lower(): c for c in df.columns}
    for c in candidates:
        hit = lowered.get(str(c).lower())
        if hit is not None:
            return hit
    return None


def _as_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("float64")


def _clip01(x: pd.Series) -> pd.Series:
    return x.clip(lower=0.0, upper=1.0)


def _normalize_percent(x: pd.Series) -> pd.Series:
    # Supports either 0..1 or 0..100
    x = _as_float(x)
    if x.dropna().empty:
        return x
    mx = float(x.quantile(0.95))
    if mx > 1.5:
        x = x / 100.0
    return _clip01(x)


def _normalize_quantile_minmax(x: pd.Series, q_low: float = 0.05, q_high: float = 0.95) -> pd.Series:
    x = _as_float(x)
    if x.dropna().empty:
        return x
    lo = float(x.quantile(q_low))
    hi = float(x.quantile(q_high))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        # Degenerate distribution -> neutral
        return pd.Series(np.full(len(x), 0.5, dtype=np.float64), index=x.index)
    return _clip01((x - lo) / (hi - lo))


def normalize_endpoint(raw: pd.Series, spec: EndpointSpec) -> pd.Series:
    """
    Normalize an ADMET endpoint into a *goodness* score in [0, 1].
    - For risk endpoints (direction=lower_better): output is safety score = 1 - normalized_risk
    """
    if spec.kind == "prob":
        x = _as_float(raw)
        # If values look like probabilities, clamp. Otherwise use robust quantile scaling.
        nonnull = x.dropna()
        if not nonnull.empty and float(nonnull.min()) >= -1e-6 and float(nonnull.max()) <= 1.0 + 1e-6:
            base = _clip01(x)
        else:
            base = _normalize_quantile_minmax(x)
    elif spec.kind == "percent":
        base = _normalize_percent(raw)
    else:
        base = _normalize_quantile_minmax(raw)

    if spec.direction == "higher_better":
        out = base
    else:
        out = 1.0 - base

    # Missing values -> neutral (0.5)
    out = out.astype("float64")
    out = out.fillna(0.5)
    return _clip01(out)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv"}:
        return pd.read_csv(path, keep_default_na=False)
    raise ValueError(f"Unsupported file type: {path} (expected .csv or .parquet)")


def build_admet_table(
    admet_df: pd.DataFrame,
    smiles_col: str,
    include_bbb: bool,
    include_dili: bool,
    include_excretion: bool,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Return a table with:
    - smiles
    - selected raw endpoints
    - selected normalized endpoints (suffix: _norm)
    """
    if smiles_col not in admet_df.columns:
        raise ValueError(f"ADMET predictions file missing smiles column: {smiles_col!r}")

    out = pd.DataFrame({"smiles": admet_df[smiles_col].astype("string")})
    used_cols: Dict[str, str] = {}

    active_specs: Iterable[EndpointSpec] = ENDPOINTS
    if not include_bbb:
        active_specs = [s for s in active_specs if s.key != "bbb"]
    if not include_dili:
        active_specs = [s for s in active_specs if s.key != "dili"]
    if not include_excretion:
        active_specs = [s for s in active_specs if s.key not in {"cl_plasma", "t_half"}]

    for spec in active_specs:
        col = _resolve_col(admet_df, spec.aliases)
        if col is None:
            logger.warning(f"ADMET endpoint not found in file: {spec.key} (aliases tried: {list(spec.aliases)[:5]}...)")
            out[spec.key] = np.nan
            out[f"{spec.key}_norm"] = 0.5
            continue
        used_cols[spec.key] = col
        out[spec.key] = admet_df[col]
        out[f"{spec.key}_norm"] = normalize_endpoint(admet_df[col], spec).astype("float32")

    return out, used_cols


def _weighted_mean(df: pd.DataFrame, cols: Sequence[str], weights: Dict[str, float]) -> pd.Series:
    acc = pd.Series(np.zeros(len(df), dtype=np.float64), index=df.index)
    wsum = 0.0
    for c in cols:
        if c not in df.columns:
            continue
        # Allow weights to be specified by endpoint key (e.g. "hia") or by column name (e.g. "hia_norm").
        base_key = c[:-5] if c.endswith("_norm") else c
        w = float(weights.get(c, weights.get(base_key, 0.0)))
        if w <= 0:
            continue
        acc = acc + w * pd.to_numeric(df[c], errors="coerce").fillna(0.5).astype("float64")
        wsum += w
    if wsum <= 0:
        return pd.Series(np.full(len(df), 0.5, dtype=np.float64), index=df.index)
    return (acc / wsum).clip(0.0, 1.0)


def compute_subscores(
    df: pd.DataFrame,
    include_bbb: bool,
    include_dili: bool,
    include_excretion: bool,
    endpoint_weights: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    # Endpoint normalized columns are "goodness" scores already.
    abs_cols = ["caco2_norm", "hia_norm", "logs_norm"]
    dist_cols = ["ppb_norm"] + (["bbb_norm"] if include_bbb else [])
    met_cols = ["cyp3a4_inh_norm", "cyp2d6_inh_norm"]
    exc_cols = ["cl_plasma_norm", "t_half_norm"] if include_excretion else []
    tox_cols = ["herg_norm", "ames_norm"] + (["dili_norm"] if include_dili else [])

    absorption_score = _weighted_mean(df, abs_cols, endpoint_weights.get("absorption", {}))
    distribution_score = _weighted_mean(df, dist_cols, endpoint_weights.get("distribution", {}))
    metabolism_score = _weighted_mean(df, met_cols, endpoint_weights.get("metabolism", {}))
    excretion_score = _weighted_mean(df, exc_cols, endpoint_weights.get("excretion", {})) if include_excretion else pd.Series(0.5, index=df.index)
    toxicity_score = _weighted_mean(df, tox_cols, endpoint_weights.get("toxicity", {}))

    out = df.copy()
    out["absorption_score"] = absorption_score.astype("float32")
    out["distribution_score"] = distribution_score.astype("float32")
    out["metabolism_score"] = metabolism_score.astype("float32")
    if include_excretion:
        out["excretion_score"] = excretion_score.astype("float32")
    out["toxicity_score"] = toxicity_score.astype("float32")
    return out


def compute_admet_score(
    df: pd.DataFrame,
    component_weights: Dict[str, float],
    include_excretion: bool,
) -> pd.Series:
    # Normalize weights over active components for stability.
    comp = dict(component_weights)
    if not include_excretion:
        comp["excretion"] = 0.0

    parts: Sequence[Tuple[str, str]] = [
        ("absorption", "absorption_score"),
        ("distribution", "distribution_score"),
        ("metabolism", "metabolism_score"),
        ("excretion", "excretion_score"),
        ("toxicity", "toxicity_score"),
    ]

    wsum = sum(float(comp.get(k, 0.0)) for k, _ in parts if float(comp.get(k, 0.0)) > 0)
    if wsum <= 0:
        return pd.Series(np.full(len(df), 0.5, dtype=np.float32), index=df.index)

    acc = pd.Series(np.zeros(len(df), dtype=np.float64), index=df.index)
    for k, col in parts:
        w = float(comp.get(k, 0.0))
        if w <= 0:
            continue
        if col not in df.columns:
            # Missing component -> neutral
            acc = acc + w * 0.5
        else:
            acc = acc + w * pd.to_numeric(df[col], errors="coerce").fillna(0.5).astype("float64")

    return (acc / wsum).clip(0.0, 1.0).astype("float32")


def minmax_01(x: pd.Series) -> pd.Series:
    x = _as_float(x)
    vmin = float(x.min(skipna=True)) if len(x) else 0.0
    vmax = float(x.max(skipna=True)) if len(x) else 1.0
    if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
        return ((x - vmin) / (vmax - vmin)).clip(0.0, 1.0).astype("float32")
    return pd.Series(np.zeros(len(x), dtype=np.float32), index=x.index)


def main() -> None:
    logger = setup_logger()

    ap = argparse.ArgumentParser(description="Step35 ADMET scoring layer (no hard filtering).")
    ap.add_argument("--input-parquet", type=Path, required=True, help="Parquet from step33/step34 (must contain smiles + qsar_prob/prob + ad_score/AD_Score).")
    ap.add_argument("--admetlab-file", type=Path, required=True, help="ADMETlab 3.0 batch prediction output file (.csv or .parquet).")
    ap.add_argument("--admet-smiles-col", type=str, default="smiles", help="SMILES column name in ADMETlab file.")
    ap.add_argument("--include-bbb", action=argparse.BooleanOptionalAction, default=True, help="Include BBB endpoint with low weight.")
    ap.add_argument("--include-dili", action=argparse.BooleanOptionalAction, default=False, help="Include DILI endpoint (optional).")
    ap.add_argument("--include-excretion", action=argparse.BooleanOptionalAction, default=False, help="Include clearance/half-life endpoints as an excretion score (optional).")
    ap.add_argument("--mode", type=str, default="default", choices=["default", "strict", "lenient"], help="Scoring mode.")
    ap.add_argument("--weights-config", type=Path, default=None, help="Optional JSON file to override component/endpoint weights.")

    ap.add_argument("--out-dir", type=Path, default=None, help="Output directory. Defaults to sibling folder next to input.")
    ap.add_argument("--out-parquet-name", type=str, default="admet_scored.parquet")
    ap.add_argument("--make-plots", action=argparse.BooleanOptionalAction, default=False, help="Write simple distribution plots (requires matplotlib).")
    args = ap.parse_args()

    if not args.input_parquet.exists():
        raise FileNotFoundError(f"Input parquet not found: {args.input_parquet}")
    if not args.admetlab_file.exists():
        raise FileNotFoundError(f"ADMETlab file not found: {args.admetlab_file}")

    if args.out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = args.input_parquet.parent / f"admet_scoring_{ts}"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading input: {args.input_parquet}")
    base = pd.read_parquet(args.input_parquet)
    base_cols = list(base.columns)  # preserve all original columns in output (and keep order)

    if "smiles" not in base.columns:
        raise ValueError("Input parquet missing 'smiles' column.")
    base["smiles"] = base["smiles"].astype("string")

    prob_col = _resolve_col(base, ["qsar_prob", "prob"])
    ad_col = _resolve_col(base, ["ad_score", "AD_Score"])
    if prob_col is None or ad_col is None:
        raise ValueError(
            "Input parquet missing required columns. "
            "Need qsar_prob/prob and ad_score/AD_Score."
        )
    if prob_col != "qsar_prob":
        base["qsar_prob"] = _as_float(base[prob_col]).astype("float32")
    if ad_col != "ad_score":
        base["ad_score"] = _as_float(base[ad_col]).astype("float32")

    logger.info(f"Loading ADMET predictions: {args.admetlab_file}")
    admet_raw = read_table(args.admetlab_file)

    component_weights, endpoint_weights = _load_weight_config(
        path=args.weights_config,
        mode=str(args.mode),
        logger=logger,
    )
    admet_tbl, used_cols = build_admet_table(
        admet_df=admet_raw,
        smiles_col=args.admet_smiles_col,
        include_bbb=bool(args.include_bbb),
        include_dili=bool(args.include_dili),
        include_excretion=bool(args.include_excretion),
        logger=logger,
    )

    # Merge by SMILES (batch tools typically use SMILES as primary key).
    # If duplicates exist, keep the first occurrence deterministically.
    admet_tbl = admet_tbl.drop_duplicates(subset=["smiles"], keep="first") ### adjust
    merged = base.merge(admet_tbl, on="smiles", how="left", validate="m:1")

    # Missing ADMET rows -> neutral defaults for normalized endpoints (already filled in build_admet_table),
    # but merge may introduce NaNs if the entire row is missing; fill those.
    norm_cols = [c for c in merged.columns if c.endswith("_norm")]
    for c in norm_cols:
        merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0.5).astype("float32")

    merged = compute_subscores(
        merged,
        include_bbb=bool(args.include_bbb),
        include_dili=bool(args.include_dili),
        include_excretion=bool(args.include_excretion),
        endpoint_weights=endpoint_weights,
    )
    merged["admet_score"] = compute_admet_score(
        merged,
        component_weights=component_weights,
        include_excretion=bool(args.include_excretion),
    ).astype("float32")

    # Final combined score (raw + pretty normalized version).
    merged["final_score_raw"] = (
        0.5 * merged["qsar_prob"].astype("float64")
        + 0.2 * merged["ad_score"].astype("float64")
        + 0.3 * merged["admet_score"].astype("float64")
    ).astype("float32")
    merged["final_score"] = minmax_01(merged["final_score_raw"])

    # Guarantee the output contains all input columns (and keep them first for readability).
    missing_from_output = [c for c in base_cols if c not in merged.columns]
    if missing_from_output:
        raise RuntimeError(f"Internal error: output is missing input columns: {missing_from_output}")
    merged = merged[base_cols + [c for c in merged.columns if c not in base_cols]]

    out_path = args.out_dir / args.out_parquet_name
    merged.to_parquet(out_path, index=False)
    logger.info(f"Saved: {out_path}")

    # Summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "input_parquet": str(args.input_parquet),
        "admetlab_file": str(args.admetlab_file),
        "admet_smiles_col": str(args.admet_smiles_col),
        "include_bbb": bool(args.include_bbb),
        "include_dili": bool(args.include_dili),
        "include_excretion": bool(args.include_excretion),
        "mode": str(args.mode),
        "weights_config": str(args.weights_config) if args.weights_config else None,
        "component_weights_used": component_weights,
        "endpoint_weights_used": endpoint_weights,
        "used_endpoint_columns": used_cols,
        "n_input": int(len(base)),
        "n_scored": int(len(merged)),
        "n_input_columns": int(len(base_cols)),
        "n_output_columns": int(len(merged.columns)),
        "final_score_raw_stats": {
            "min": float(np.nanmin(merged["final_score_raw"])),
            "mean": float(np.nanmean(merged["final_score_raw"])),
            "max": float(np.nanmax(merged["final_score_raw"])),
        },
        "final_score_stats": {
            "min": float(np.nanmin(merged["final_score"])),
            "mean": float(np.nanmean(merged["final_score"])),
            "max": float(np.nanmax(merged["final_score"])),
        },
        "admet_score_stats": {
            "min": float(np.nanmin(merged["admet_score"])),
            "mean": float(np.nanmean(merged["admet_score"])),
            "max": float(np.nanmax(merged["admet_score"])),
        },
    }
    (args.out_dir / "admet_scoring_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"Saved summary: {args.out_dir / 'admet_scoring_summary.json'}")

    if args.make_plots:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            axes[0].hist(merged["admet_score"].to_numpy(), bins=50, color="#2c7fb8", alpha=0.9)
            axes[0].set_title("admet_score")
            axes[1].hist(merged["final_score_raw"].to_numpy(), bins=50, color="#7fcdbb", alpha=0.9)
            axes[1].set_title("final_score_raw")
            axes[2].hist(merged["final_score"].to_numpy(), bins=50, color="#f03b20", alpha=0.9)
            axes[2].set_title("final_score (normalized)")
            for ax in axes:
                ax.set_xlabel("value")
                ax.set_ylabel("count")
            fig.tight_layout()
            fig_path = args.out_dir / "admet_score_distributions.png"
            fig.savefig(fig_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Saved plot: {fig_path}")
        except Exception as exc:
            logger.warning(f"Plotting skipped: {exc}")


if __name__ == "__main__":
    main()


# %%
# Plotting-only cell (interactive)
##############################################################################
# Goal:
#   - Load the saved parquet from a completed Step35 run
#   - Plot score distributions without recomputation
#   - Export PNG + SVG with configurable fonts
##############################################################################
try:
    from IPython import get_ipython  # type: ignore

    _IN_IPYTHON = get_ipython() is not None
except Exception:
    _IN_IPYTHON = False

if _IN_IPYTHON:
    from pathlib import Path

    import pandas as pd
    import matplotlib.pyplot as plt

    # --- USER EDITABLE ---
    # Point this to the output directory created by this script (admet_scoring_YYYYMMDD_HHMMSS)
    OUT_DIR = Path("../models_out/qsar_ml_20260412_162829/virtual_screening/ad_screening_results_20260419_140044/admet_scoring_20260419_173114/")
    OUT_PARQUET = OUT_DIR / "admet_scored.parquet"
    # ---------------------

    if not OUT_PARQUET.exists():
        raise FileNotFoundError(f"Missing parquet: {OUT_PARQUET.resolve()}")

    PLOT_STYLE = {
        # If your machine does not have the font, matplotlib will fallback automatically.
        "font.family": "serif",
        "font.serif": ["Cambria", "Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.linewidth": 1.1,
        "grid.alpha": 0.25,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }

    plt.rcParams.update(PLOT_STYLE)

    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(OUT_PARQUET)

    def _save(fig, name: str) -> None:
        for ext in ["png", "svg"]:
            fig.savefig(fig_dir / f"{name}.{ext}", bbox_inches="tight")
        plt.show()

    # A. Distributions
    cols = [c for c in ["qsar_prob", "ad_score", "admet_score", "final_score_raw", "final_score"] if c in df.columns]
    n = len(cols)
    if n:
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
        if n == 1:
            axes = [axes]
        for ax, col in zip(axes, cols):
            ax.hist(pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(), bins=60, color="#2c7fb8", alpha=0.9)
            ax.set_title(col)
            ax.set_xlabel("value")
            ax.set_ylabel("count")
            ax.grid(True, linestyle=":", linewidth=0.8)
        fig.tight_layout()
        _save(fig, "admet_step35_distributions")

    # B. Pairwise scatter: final_score vs each component score
    if "final_score" in df.columns:
        for col in ["qsar_prob", "ad_score", "admet_score"]:
            if col not in df.columns:
                continue
            fig, ax = plt.subplots(figsize=(5.5, 5))
            x = pd.to_numeric(df[col], errors="coerce")
            y = pd.to_numeric(df["final_score"], errors="coerce")
            ax.scatter(x, y, s=8, alpha=0.35, c="#7fcdbb", edgecolors="none")
            ax.set_xlabel(col)
            ax.set_ylabel("final_score")
            ax.set_title(f"final_score vs {col}")
            ax.grid(True, linestyle=":", linewidth=0.8)
            fig.tight_layout()
            _save(fig, f"final_vs_{col}")

# %%
