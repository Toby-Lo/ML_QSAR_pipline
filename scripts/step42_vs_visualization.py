#!/usr/bin/env python3
"""
Step42: Visualize hierarchical virtual screening funnel and score distributions.

Main output:
  - FigureX_vs_funnel_ABCD.svg
    (A) Funnel attrition across screening stages
    (B) Calibrated QSAR probability distribution
    (C) AD score distribution
    (D) Final score distribution for prioritized compounds

Supplementary outputs:
  - FigureS_vs_prob_vs_ad_hexbin.svg
  - FigureS_vs_stage_retention.svg
  - vs_funnel_stage_counts.csv
  - vs_distribution_summary.csv

Example:
python scripts/step42_vs_visualization.py \
  --run-dir models_out/qsar_ml_20260412_162829 \
  --vs-dir models_out/qsar_ml_20260412_162829/virtual_screening \
  --ad-screening-dir models_out/qsar_ml_20260412_162829/virtual_screening/ad_screening_results_20260419_140044 \
  --admet-parquet models_out/qsar_ml_20260412_162829/virtual_screening/ad_screening_results_20260419_140044/admet_scoring_20260419_173114/admet_scored.parquet \
  --output-dir models_out/qsar_ml_20260412_162829/virtual_screening/figures_vs

--ad-threshold 0.30
--ad-xmin 0.29
--ad-xmax 0.46
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"Missing dependency: pandas ({exc})")

try:
    import pyarrow.parquet as pq
except Exception:
    pq = None

from matplotlib import pyplot as plt


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Virtual screening funnel visualization (step42).")
    p.add_argument("--run-dir", type=Path, default=Path("models_out/qsar_ml_20260412_162829"))
    p.add_argument("--vs-dir", type=Path, default=None, help="Virtual screening root dir. Default: <run-dir>/virtual_screening")
    p.add_argument("--ad-screening-dir", type=Path, default=None, help="Path to step34 output dir containing ranked_results.parquet")
    p.add_argument("--admet-parquet", type=Path, default=None, help="Optional step35 output parquet (contains final_score).")
    p.add_argument("--db-all", type=Path, default=Path("data/database/all_zinc_combined.parquet"))
    p.add_argument("--db-filtered", type=Path, default=Path("data/database/zinc_filtered.parquet"))
    p.add_argument("--db-druglike", type=Path, default=Path("data/database/zinc_druglike.parquet"))
    p.add_argument("--db-features", type=Path, default=Path("data/database/zinc_features.parquet"))
    p.add_argument("--bins", type=int, default=40)
    p.add_argument("--ad-threshold", type=float, default=0.30, help="AD threshold line for panel C.")
    p.add_argument("--ad-xmin", type=float, default=0.29, help="Panel C x-axis min.")
    p.add_argument("--ad-xmax", type=float, default=0.46, help="Panel C x-axis max.")
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args(argv)


def _count_rows(path: Optional[Path]) -> Optional[int]:
    if path is None or not path.exists():
        return None
    if path.suffix.lower() == ".parquet":
        if pq is not None:
            try:
                return int(pq.ParquetFile(path).metadata.num_rows)
            except Exception:
                pass
        return int(len(pd.read_parquet(path, columns=[])))
    if path.suffix.lower() == ".csv":
        return int(len(pd.read_csv(path)))
    return None


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def _resolve_latest_ad_screening_dir(vs_dir: Path) -> Optional[Path]:
    cand = sorted(vs_dir.glob("ad_screening_results_*"), key=lambda p: p.name)
    return cand[-1] if cand else None


def _resolve_latest_prediction_parquet(vs_dir: Path) -> Optional[Path]:
    cand = sorted(vs_dir.glob("zinc_predictions_*.parquet"), key=lambda p: p.name)
    if cand:
        return cand[-1]
    nested = sorted(vs_dir.glob("results_*/zinc_predictions_*.parquet"), key=lambda p: p.name)
    return nested[-1] if nested else None


def _resolve_latest_admet_parquet(ad_screening_dir: Path) -> Optional[Path]:
    candidates = [
        ad_screening_dir / "top1000_full_info.parquet",
        ad_screening_dir / "top_candidates.parquet",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _pick_col(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    cols = set(df.columns)
    for n in names:
        if n in cols:
            return n
    return None


def _series_float(df: pd.DataFrame, names: Sequence[str]) -> Optional[np.ndarray]:
    c = _pick_col(df, names)
    if c is None:
        return None
    return pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64)


def _format_stage_label(stage: str) -> str:
    mapping = {
        "Initial library": "Initial\nlibrary",
        "PhysChem/PAINS filter": "PhysChem/PAINS\nfilter",
        "Strict drug-like filter": "Strict drug-like\nfilter",
        "Feature-ready compounds": "Feature-ready\ncompounds",
        "QSAR + AD scored": "QSAR + AD\nscored",
        "AD-aware filtered": "AD-aware\nfiltered",
        "Prioritized candidates": "Prioritized\ncandidates",
    }
    return mapping.get(stage, stage)


def _split_stage_label(stage: str) -> Tuple[str, str]:
    lab = _format_stage_label(stage)
    parts = lab.split("\n", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return lab, ""


def build_stage_table(
    db_all: Path,
    db_filtered: Path,
    db_druglike: Path,
    db_features: Path,
    pred_path: Optional[Path],
    ranked_path: Optional[Path],
    top_path: Optional[Path],
) -> pd.DataFrame:
    stages: List[Tuple[str, Optional[Path]]] = [
        ("Initial library", db_all),
        ("PhysChem/PAINS filter", db_filtered),
        ("Strict drug-like filter", db_druglike),
        ("Feature-ready compounds", db_features),
        ("QSAR + AD scored", pred_path),
        ("AD-aware filtered", ranked_path),
        ("Prioritized candidates", top_path),
    ]
    rows: List[Dict[str, object]] = []
    prev_n: Optional[int] = None
    first_n: Optional[int] = None
    for stage, path in stages:
        n = _count_rows(path)
        if first_n is None and n is not None:
            first_n = n
        retain_prev = (float(n) / prev_n) if (n is not None and prev_n not in (None, 0)) else np.nan
        retain_initial = (float(n) / first_n) if (n is not None and first_n not in (None, 0)) else np.nan
        rows.append(
            {
                "stage": stage,
                "path": str(path) if path is not None else "",
                "n_compounds": n,
                "retention_vs_prev": retain_prev,
                "retention_vs_initial": retain_initial,
            }
        )
        if n is not None:
            prev_n = n
    return pd.DataFrame(rows)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve()
    vs_dir = (args.vs_dir or (run_dir / "virtual_screening")).expanduser().resolve()

    ad_screening_dir = args.ad_screening_dir.expanduser().resolve() if args.ad_screening_dir else _resolve_latest_ad_screening_dir(vs_dir)
    if ad_screening_dir is None or not ad_screening_dir.exists():
        raise SystemExit("Cannot resolve step34 output dir. Set --ad-screening-dir explicitly.")

    pred_path = _resolve_latest_prediction_parquet(vs_dir)
    ranked_path = ad_screening_dir / "ranked_results.parquet"
    top_path = ad_screening_dir / "top_candidates.parquet"
    admet_path = args.admet_parquet.expanduser().resolve() if args.admet_parquet else _resolve_latest_admet_parquet(ad_screening_dir)

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else (ad_screening_dir / f"vs_figures_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_df = build_stage_table(
        db_all=args.db_all.expanduser().resolve(),
        db_filtered=args.db_filtered.expanduser().resolve(),
        db_druglike=args.db_druglike.expanduser().resolve(),
        db_features=args.db_features.expanduser().resolve(),
        pred_path=pred_path,
        ranked_path=ranked_path if ranked_path.exists() else None,
        top_path=top_path if top_path.exists() else None,
    )
    stage_df.to_csv(output_dir / "vs_funnel_stage_counts.csv", index=False)

    if not ranked_path.exists():
        raise SystemExit(f"Missing ranked results parquet: {ranked_path}")
    ranked_df = _read_table(ranked_path)

    top_df = _read_table(top_path) if top_path.exists() else ranked_df.head(min(1000, len(ranked_df))).copy()
    final_df = _read_table(admet_path) if (admet_path is not None and admet_path.exists()) else top_df

    prob = _series_float(ranked_df, ["qsar_prob", "prob"])
    ad_score = _series_float(ranked_df, ["ad_score", "AD_Score"])
    final_score = _series_float(final_df, ["final_score", "qsar_ad_rank_score", "qsar_ad_rank_score_raw"])

    if prob is None:
        raise SystemExit("Cannot find probability column in ranked results (expected qsar_prob/prob).")
    if ad_score is None:
        raise SystemExit("Cannot find AD score column in ranked results (expected ad_score/AD_Score).")
    if final_score is None:
        raise SystemExit("Cannot find final score column in top/final results.")

    valid_prob = prob[np.isfinite(prob)]
    valid_ad = ad_score[np.isfinite(ad_score)]
    valid_final = final_score[np.isfinite(final_score)]

    dist_rows = [
        {"metric": "qsar_prob", "n": len(valid_prob), "mean": float(np.mean(valid_prob)), "std": float(np.std(valid_prob)), "min": float(np.min(valid_prob)), "max": float(np.max(valid_prob))},
        {"metric": "ad_score", "n": len(valid_ad), "mean": float(np.mean(valid_ad)), "std": float(np.std(valid_ad)), "min": float(np.min(valid_ad)), "max": float(np.max(valid_ad))},
        {"metric": "final_score", "n": len(valid_final), "mean": float(np.mean(valid_final)), "std": float(np.std(valid_final)), "min": float(np.min(valid_final)), "max": float(np.max(valid_final))},
    ]
    pd.DataFrame(dist_rows).to_csv(output_dir / "vs_distribution_summary.csv", index=False)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Cambria", "Times New Roman", "DejaVu Serif", "Times"],
            "font.size": 10.5,
            "axes.grid": False,
            "svg.fonttype": "none",
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), constrained_layout=True)

    # A) Funnel
    ax = axes[0, 0]
    use = stage_df.dropna(subset=["n_compounds"]).copy()
    # "QSAR + AD scored" is nearly identical to "Feature-ready compounds" in most runs
    # (often differs by ~1 row due to invalid/missing entries), so hide it for cleaner funnel.
    use = use[use["stage"] != "QSAR + AD scored"].copy()
    # "Feature-ready compounds" is also often identical to "Strict drug-like filter" after featurization.
    use = use[use["stage"] != "Feature-ready compounds"].copy()
    y = np.arange(len(use))
    vals = use["n_compounds"].astype(float).to_numpy()
    vals_million = vals / 1_000_000.0
    ax.barh(y, vals_million, color="#4c78a8", alpha=0.88)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    x_label = -0.025
    dy = 0.15
    for yi, s in zip(y, use["stage"].tolist()):
        line1, line2 = _split_stage_label(s)
        ax.text(
            x_label,
            yi - dy,
            line1,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=10.5,
            color="black",
        )
        if line2:
            ax.text(
                x_label,
                yi + dy,
                line2,
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=10.5,
                color="black",
            )
    ax.invert_yaxis()
    ax.set_xlabel("Compound count (×10^6)")
    ax.set_title("(A) Hierarchical Screening Funnel")
    x_left = float(np.max(vals_million)) * 0.015 if len(vals_million) else 0.01
    for i, v_raw in enumerate(vals):
        ax.text(x_left, i, f"{int(v_raw):,}", va="center", ha="left", fontsize=9, color="black")

    # B) QSAR probability
    ax = axes[0, 1]
    ax.hist(valid_prob, bins=args.bins, color="#1f77b4", alpha=0.75, density=True, edgecolor="white", linewidth=0.4)
    ax.axvline(float(np.mean(valid_prob)), linestyle="--", color="black", linewidth=1.0, label="Mean")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Calibrated QSAR probability")
    ax.set_ylabel("Density")
    ax.set_title("(B) QSAR Probability Distribution")
    ax.legend(loc="best", fontsize=8)

    # C) AD score
    ax = axes[1, 0]
    ax.hist(valid_ad, bins=args.bins, color="#e15759", alpha=0.72, density=True, edgecolor="white", linewidth=0.4)
    ax.axvline(float(np.mean(valid_ad)), linestyle="--", color="black", linewidth=1.0, label="Mean")
    ax.axvline(float(args.ad_threshold), linestyle=":", color="#8b0000", linewidth=1.2, label=f"AD threshold = {args.ad_threshold:.2f}")
    ax.set_xlim(float(args.ad_xmin), float(args.ad_xmax))
    ax.set_xlabel("AD score")
    ax.set_ylabel("Density")
    ax.set_title("(C) AD Score Distribution")
    ax.legend(loc="best", fontsize=8)

    # D) final score
    ax = axes[1, 1]
    ax.hist(valid_final, bins=args.bins, color="#59a14f", alpha=0.78, density=True, edgecolor="white", linewidth=0.4)
    ax.axvline(float(np.mean(valid_final)), linestyle="--", color="black", linewidth=1.0, label="Mean")
    ax.set_xlabel("Final score")
    ax.set_ylabel("Density")
    ax.set_title("(D) Final Priority Score Distribution")
    ax.legend(loc="best", fontsize=8)

    out_main = output_dir / "FigureX_vs_funnel_ABCD.svg"
    fig.savefig(out_main, format="svg", bbox_inches="tight")
    plt.close(fig)

    # Supplementary 1: Prob vs AD hexbin
    fig, ax = plt.subplots(figsize=(6.0, 4.8), constrained_layout=True)
    mask = np.isfinite(prob) & np.isfinite(ad_score)
    hb = ax.hexbin(prob[mask], ad_score[mask], gridsize=35, cmap="viridis", mincnt=1)
    fig.colorbar(hb, ax=ax, label="Count")
    ax.axhline(
        float(args.ad_threshold),
        linestyle=":",
        color="#8b0000",
        linewidth=1.2,
        label=f"AD threshold = {args.ad_threshold:.2f}",
    )
    ax.set_xlabel("Calibrated QSAR probability")
    ax.set_ylabel("AD score")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(float(args.ad_xmin), float(args.ad_xmax))
    ax.set_title("Supplementary: Probability vs AD Score")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.savefig(output_dir / "FigureS_vs_prob_vs_ad_hexbin.svg", format="svg", bbox_inches="tight")
    plt.close(fig)

    # Supplementary 2: Retention curve
    fig, ax1 = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
    x = np.arange(len(use))
    retain_prev = pd.to_numeric(use["retention_vs_prev"], errors="coerce").to_numpy(dtype=np.float64)
    retain_init = pd.to_numeric(use["retention_vs_initial"], errors="coerce").to_numpy(dtype=np.float64)
    ax1.bar(x, retain_prev * 100.0, color="#f28e2b", alpha=0.72, label="Retention vs previous (%)")
    ax1.set_ylabel("Retention vs previous (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(use["stage"].tolist(), rotation=25, ha="right")
    ax2 = ax1.twinx()
    ax2.plot(x, retain_init * 100.0, color="black", marker="o", linewidth=1.6, label="Retention vs initial (%)")
    ax2.set_ylabel("Retention vs initial (%)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8, frameon=False)
    ax1.set_title("Supplementary: Stage Retention Profile")
    fig.savefig(output_dir / "FigureS_vs_stage_retention.svg", format="svg", bbox_inches="tight")
    plt.close(fig)

    meta = {
        "run_dir": str(run_dir),
        "vs_dir": str(vs_dir),
        "ad_screening_dir": str(ad_screening_dir),
        "prediction_file": str(pred_path) if pred_path is not None else None,
        "ranked_file": str(ranked_path),
        "top_file": str(top_path) if top_path.exists() else None,
        "admet_file": str(admet_path) if admet_path is not None else None,
        "output_dir": str(output_dir),
        "main_figure": str(out_main),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (output_dir / "vs_visualization_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("[OK] VS visualization complete")
    print(f"  - Output dir: {output_dir}")
    print(f"  - Main figure: {out_main.name}")
    print("  - Supplementary: FigureS_vs_prob_vs_ad_hexbin.svg, FigureS_vs_stage_retention.svg")
    print("  - Tables: vs_funnel_stage_counts.csv, vs_distribution_summary.csv")


if __name__ == "__main__":
    main()
