"""Plot threshold analysis figures from step10_qsar_ml.py outputs.

Usage examples:
    python scripts/step41_threshold_analysis.py
    python scripts/step41_threshold_analysis.py --base-dir models_out/qsar_ml_YYYYMMDD_HHMMSS
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot threshold analysis from step10 outputs")
    parser.add_argument("--base-dir", type=Path, help="Run directory (e.g. models_out/qsar_ml_YYYYMMDD_HHMMSS)")
    parser.add_argument("--output-dir", type=Path, help="Figure output directory (default: <base-dir>/figures/threshold_analysis)")
    parser.add_argument("--palette", default="colorblind", help="Seaborn palette name")
    parser.add_argument("--dpi", type=int, default=600, help="Figure DPI")
    parser.add_argument("--font", default="Cambria", help="Serif font for publication style")
    return parser.parse_args()


def configure_plotting(font: str) -> None:
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = font
    rcParams["text.usetex"] = False
    rcParams["axes.edgecolor"] = "black"
    rcParams["axes.linewidth"] = 1.2
    rcParams["axes.facecolor"] = "white"
    rcParams["grid.color"] = "white"
    rcParams["figure.facecolor"] = "white"
    rcParams["savefig.facecolor"] = "white"


def resolve_base_dir(base_dir: Optional[Path]) -> Path:
    if base_dir is not None:
        if not base_dir.exists():
            raise FileNotFoundError(f"Base directory not found: {base_dir}")
        return base_dir
    models_out = Path("models_out")
    candidates = sorted([p for p in models_out.glob("qsar_ml_*") if p.is_dir()])
    if not candidates:
        raise FileNotFoundError("No run folder found under models_out (pattern: qsar_ml_*)")
    return candidates[-1]


def split_seed_dirs(base_dir: Path) -> List[Path]:
    return sorted([p for p in base_dir.glob("split_seed_*") if p.is_dir()])


def parse_seed_from_name(split_dir: Path) -> Optional[int]:
    try:
        return int(split_dir.name.split("_")[-1])
    except ValueError:
        return None


def load_threshold_data(base_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    curves_frames: List[pd.DataFrame] = []
    selection_frames: List[pd.DataFrame] = []
    for split_dir in split_seed_dirs(base_dir):
        seed = parse_seed_from_name(split_dir)
        results_dir = split_dir / "results"
        curve_path = results_dir / "threshold_curves_data.csv"
        summary_path = results_dir / "threshold_selection_summary.csv"
        if curve_path.exists():
            cdf = pd.read_csv(curve_path)
            if "split_seed" not in cdf.columns:
                cdf["split_seed"] = seed
            curves_frames.append(cdf)
        if summary_path.exists():
            sdf = pd.read_csv(summary_path)
            if "split_seed" not in sdf.columns:
                sdf["split_seed"] = seed
            selection_frames.append(sdf)
    curves_df = pd.concat(curves_frames, ignore_index=True) if curves_frames else pd.DataFrame()
    selection_df = pd.concat(selection_frames, ignore_index=True) if selection_frames else pd.DataFrame()
    return curves_df, selection_df


def nearest_row(df: pd.DataFrame, threshold: float) -> Optional[pd.Series]:
    if df.empty or "Threshold" not in df.columns:
        return None
    idx = (df["Threshold"].astype(float) - float(threshold)).abs().idxmin()
    return df.loc[idx]


def plot_model_threshold_panel(seed: int,
                               model: str,
                               model_df: pd.DataFrame,
                               summary_row: Optional[pd.Series],
                               output_dir: Path,
                               color: Tuple[float, float, float],
                               dpi: int,
                               font: str) -> Optional[Tuple[Path, Path]]:
    oof_df = model_df[model_df["dataset"] == "oof"].copy()
    ext_df = model_df[model_df["dataset"] == "external"].copy()
    if oof_df.empty:
        return None

    configure_plotting(font)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    # ===== Panel 1: ROC & Youden's J =====
    oof_roc = oof_df[["FPR", "TPR"]].dropna().sort_values("FPR")
    axes[0].plot(oof_roc["FPR"], oof_roc["TPR"], color=color, linewidth=1.8, label="OOF ROC")
    if not ext_df.empty:
        ext_roc = ext_df[["FPR", "TPR"]].dropna().sort_values("FPR")
        axes[0].plot(ext_roc["FPR"], ext_roc["TPR"], color="black", linestyle="--", linewidth=1.2, alpha=0.7, label="External ROC")
    axes[0].plot([0, 1], [0, 1], linestyle=":", color="gray", linewidth=1.0)

    if summary_row is not None and pd.notna(summary_row.get("youden_j_threshold")):
        j_thr = float(summary_row["youden_j_threshold"])
        j_row = nearest_row(oof_df, j_thr)
        if j_row is not None:
            axes[0].scatter([float(j_row["FPR"])], [float(j_row["TPR"])], s=40, color="crimson", zorder=5, label=f"Youden-J thr={j_thr:.3f}")

    axes[0].set_title("ROC & Youden's J", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(loc="lower right", frameon=False, fontsize=8)

    # ===== Panel 2: PR with multiple threshold points =====
    oof_pr = oof_df[["Recall", "Precision"]].dropna().sort_values("Recall")
    axes[1].plot(oof_pr["Recall"], oof_pr["Precision"], color=color, linewidth=1.8, label="OOF PR")
    if not ext_df.empty:
        ext_pr = ext_df[["Recall", "Precision"]].dropna().sort_values("Recall")
        axes[1].plot(ext_pr["Recall"], ext_pr["Precision"], color="black", linestyle="--", linewidth=1.2, alpha=0.7, label="External PR")

    # Plot multiple threshold points on PR curve
    threshold_configs = [
        ("max_f1_threshold", "Max-F1", "crimson", "o"),
        ("max_precision_threshold", "Max-Prec", "blue", "s"),
        ("max_recall_threshold", "Max-Recall", "green", "^"),
    ]
    for thr_col, thr_label, thr_color, marker in threshold_configs:
        if summary_row is not None and pd.notna(summary_row.get(thr_col)):
            thr = float(summary_row[thr_col])
            thr_row = nearest_row(oof_df, thr)
            if thr_row is not None:
                axes[1].scatter([float(thr_row["Recall"])], [float(thr_row["Precision"])], 
                               s=30, color=thr_color, marker=marker, zorder=5, label=f"{thr_label} thr={thr:.3f}")

    axes[1].set_title("PR Curve & Multi-Metric Thresholds", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(loc="lower left", frameon=False, fontsize=7.5)

    # ===== Panel 3: F1 & MCC vs Threshold =====
    oof_thr = oof_df[["Threshold", "F1"]].dropna().sort_values("Threshold")
    axes[2].plot(oof_thr["Threshold"], oof_thr["F1"], color=color, linewidth=1.8, label="OOF F1")
    if not ext_df.empty:
        ext_thr = ext_df[["Threshold", "F1"]].dropna().sort_values("Threshold")
        axes[2].plot(ext_thr["Threshold"], ext_thr["F1"], color="black", linestyle="--", linewidth=1.3, alpha=0.7, label="External F1")

    # Also plot MCC if available
    if "MCC" in oof_df.columns:
        oof_mcc = oof_df[["Threshold", "MCC"]].dropna().sort_values("Threshold")
        axes[2].plot(oof_mcc["Threshold"], oof_mcc["MCC"], color=color, linewidth=1.2, 
                    linestyle="--", alpha=0.6, label="OOF MCC")

    # Mark selected/optimal thresholds
    threshold_vlines = [
        ("max_f1_threshold", "Max-F1", "crimson", "-"),
        ("max_mcc_threshold", "Max-MCC", "blue", "--"),
        ("youden_j_threshold", "Youden-J", "green", ":"),
    ]
    for thr_col, thr_label, thr_color, linestyle in threshold_vlines:
        if summary_row is not None and pd.notna(summary_row.get(thr_col)):
            thr = float(summary_row[thr_col])
            axes[2].axvline(thr, color=thr_color, linestyle=linestyle, linewidth=1.3, 
                           alpha=0.7, label=f"{thr_label}={thr:.3f}")

    axes[2].set_title("F1 & MCC vs Threshold", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Threshold")
    axes[2].set_ylabel("Score")
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].legend(loc="best", frameon=False, fontsize=7.5)

    for ax in axes:
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"Threshold Analysis | Seed={seed} | Model={model}", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_model = str(model).replace("/", "_").replace(" ", "_")
    png_path = output_dir / f"seed_{seed}_{safe_model}_threshold_analysis.png"
    svg_path = output_dir / f"seed_{seed}_{safe_model}_threshold_analysis.svg"
    fig.savefig(png_path, dpi=dpi, format="png")
    fig.savefig(svg_path, dpi=dpi, format="svg")
    plt.close(fig)
    return png_path, svg_path


def main() -> None:
    args = parse_args()
    base_dir = resolve_base_dir(args.base_dir)
    output_dir = args.output_dir or (base_dir / "figures" / "threshold_analysis")

    print(f"[INFO] Base run directory: {base_dir}")
    print(f"[INFO] Figures will be saved to: {output_dir}")

    curves_df, selection_df = load_threshold_data(base_dir)
    if curves_df.empty:
        print("[WARN] No threshold_curves_data.csv found under split_seed_*/results.")
        return

    # Core required columns (some optional metrics may be missing)
    required_cols = {"split_seed", "model", "dataset", "Threshold", "TPR", "FPR"}
    missing = required_cols - set(curves_df.columns)
    if missing:
        raise ValueError(f"threshold_curves_data.csv missing core columns: {sorted(missing)}")
    
    # Optional metrics (only warn if none are present)
    optional_metrics = {"Precision", "Recall", "F1", "MCC"}
    available_metrics = optional_metrics & set(curves_df.columns)
    if not available_metrics:
        print("[WARN] threshold_curves_data.csv has no metric columns (Precision, Recall, F1, MCC).")

    models = sorted(curves_df["model"].astype(str).unique().tolist())
    palette = sns.color_palette(args.palette, n_colors=max(len(models), 3))
    color_map = {model: palette[idx % len(palette)] for idx, model in enumerate(models)}

    saved = 0
    for seed in sorted(curves_df["split_seed"].dropna().astype(int).unique().tolist()):
        seed_df = curves_df[curves_df["split_seed"].astype(int) == int(seed)]
        for model in sorted(seed_df["model"].astype(str).unique().tolist()):
            model_df = seed_df[seed_df["model"].astype(str) == model].copy()
            summary_row = None
            if not selection_df.empty:
                sub = selection_df[
                    (selection_df["split_seed"].astype(int) == int(seed))
                    & (selection_df["model"].astype(str) == model)
                ]
                if not sub.empty:
                    summary_row = sub.iloc[0]
            out = plot_model_threshold_panel(
                seed=int(seed),
                model=model,
                model_df=model_df,
                summary_row=summary_row,
                output_dir=output_dir,
                color=color_map[model],
                dpi=args.dpi,
                font=args.font,
            )
            if out is not None:
                png_path, svg_path = out
                print(f"[OK] Saved: {png_path}")
                print(f"[OK] Saved: {svg_path}")
                saved += 1

    if saved == 0:
        print("[WARN] No figures were generated. Check threshold curve files and model columns.")


if __name__ == "__main__":
    main()
