"""Plot QSAR performance figures from step10_qsar_ml.py outputs.

Usage examples:
    python scripts/step40_plot_performance.py
    python scripts/step40_plot_performance.py --base-dir models_out/qsar_ml_20260409_122600
    python scripts/step40_plot_performance.py --include-external --include-cv --boxplot-stage both

python scripts/step40_plot_performance.py \
  --base-dir models_out/qsar_ml_20260412_162829 \
  --include-external \
  --include-cv \
  --boxplot-stage both \
  --error-type std \
  --output-dir models_out/qsar_ml_20260412_162829/figures/performance/ \
  --polar-show-values 

  
optional arguments:
--boxplot-metrics mcc,f1,accuracy,roc_auc,pr_auc
--palette colorblind
--dpi 600
--font Cambria
"""

from __future__ import annotations

# %%
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
from matplotlib.patches import Circle, Patch
from sklearn.metrics import auc, precision_recall_curve, roc_curve, average_precision_score, roc_auc_score


# %%
DEFAULT_METRICS = ["mcc", "f1", "accuracy", "precision", "recall"]
METRIC_ALIASES = {
    "acc": "accuracy",
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "mcc": "mcc",
    "auc": "roc_auc",
    "roc_auc": "roc_auc",
    "pr_auc": "pr_auc",
}
METRIC_LABELS = {
    "accuracy": "ACC",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "mcc": "MCC",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
}
BOXPLOT_ALLOWED_METRICS = {"mcc", "f1", "accuracy", "precision", "recall"}
POLAR_METRIC_ORDER = ["mcc", "f1", "accuracy", "precision", "recall"]
CI_Z = 1.96
LEGEND_DECIMALS = 3
MODEL_COLORS = {
    'ETC': '#B08B86',   # 莫兰迪肉粉/浅褐
    'RFC': '#C3B083',   # 暗金/莫兰迪淡黄
    'SVC': '#1F77B4',   # 核心高亮蓝色
    'MLP': '#8F93B5',   # 莫兰迪紫灰/蓝灰
    'XGBC': '#8AA68A',  # 莫兰迪豆绿/灰绿
    'LR': '#A8829A'     # 莫兰迪暗紫/粉灰
}


# %%
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot QSAR metrics from step10 outputs")
    parser.add_argument("--base-dir", type=Path, help="Run directory (e.g. models_out/qsar_ml_YYYYMMDD_HHMMSS)")
    parser.add_argument("--output-dir", type=Path, help="Figure output directory (default: <base-dir>/figures)")
    parser.add_argument("--include-external", action="store_true", help="Plot external ROC/PR")
    parser.add_argument("--include-cv", action="store_true", help="Plot CV ROC/PR")
    parser.add_argument(
        "--boxplot-stage",
        choices=["external", "cv", "both", "none"],
        default="both",
        help="Which stage(s) to draw metric boxplots for",
    )
    parser.add_argument(
        "--boxplot-metrics",
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated boxplot metrics (aliases allowed; default: mcc,f1,accuracy,precision,recall)",
    )
    parser.add_argument("--palette", default="colorblind", help="Seaborn palette name")
    parser.add_argument("--dpi", type=int, default=600, help="Figure DPI")
    parser.add_argument("--font", default="Cambria", help="Serif font for publication style")
    parser.add_argument("--polar-show-values", action="store_true", help="Show numeric values above polar bars")
    parser.add_argument(
        "--error-type",
        choices=["std", "ci95"],
        default="std",
        help="Error type for plots (std or ci95)",
    )
    return parser.parse_args()


# %%
def configure_plotting(font: str) -> None:
    # Prefer Cambria for publication-style figures while keeping robust fallbacks.
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = [font, "Cambria", "Times New Roman", "DejaVu Serif"]
    rcParams["text.usetex"] = False
    rcParams["font.size"] = 13
    rcParams["axes.titlesize"] = 15
    rcParams["axes.labelsize"] = 13
    rcParams["xtick.labelsize"] = 12
    rcParams["ytick.labelsize"] = 12
    rcParams["legend.fontsize"] = 11
    rcParams["axes.edgecolor"] = "black"
    rcParams["axes.linewidth"] = 1.2
    rcParams["axes.facecolor"] = "white"
    rcParams["axes.spines.top"] = True
    rcParams["axes.spines.right"] = True
    rcParams["grid.color"] = "white"
    rcParams["figure.facecolor"] = "white"
    rcParams["savefig.facecolor"] = "white"
    rcParams["savefig.bbox"] = "tight"
    rcParams["xtick.direction"] = "in"
    rcParams["ytick.direction"] = "in"
    rcParams["xtick.major.width"] = 1.0
    rcParams["ytick.major.width"] = 1.0
    rcParams["xtick.major.size"] = 5
    rcParams["ytick.major.size"] = 5
    rcParams["legend.borderpad"] = 0.5
    rcParams["legend.handlelength"] = 1.8


# %%
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


# %%
def split_seed_dirs(base_dir: Path) -> List[Path]:
    return sorted([p for p in base_dir.glob("split_seed_*") if p.is_dir()])


def collect_prediction_files(base_dir: Path, stage: str) -> List[Path]:
    files: List[Path] = []
    for split_dir in split_seed_dirs(base_dir):
        pred_dir = split_dir / "predictions"
        if stage == "external":
            candidate = pred_dir / "external_test_predictions.csv"
            if candidate.exists():
                files.append(candidate)
        elif stage == "cv":
            files.extend(sorted(pred_dir.glob("cv_predictions_fold_*.csv")))
    return files


# %%
def interpolate_curve(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.interp(grid, x, y, left=y[0], right=y[-1])


# %%
def error_from_samples(samples: List[np.ndarray], error_type: str = "ci95") -> np.ndarray:
    arr = np.asarray(samples)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[0] <= 1:
        return np.zeros(arr.shape[1], dtype=float)
    if error_type == "std":
        # Match pipeline CSV summaries which use population std (ddof=0).
        return np.std(arr, axis=0, ddof=0)
    sem = np.std(arr, axis=0, ddof=0) / np.sqrt(arr.shape[0])
    return CI_Z * sem


def error_from_scalars(values: List[float], error_type: str = "ci95") -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size <= 1:
        return 0.0
    if error_type == "std":
        # Match pipeline CSV summaries which use population std (ddof=0).
        return float(np.std(arr, ddof=0))
    sem = np.std(arr, ddof=0) / np.sqrt(arr.size)
    return float(CI_Z * sem)


def get_or_assign_model_colors(models: List[str], palette_name: str, color_map: Dict[str, tuple]) -> None:
    """Assign colors to models from predefined MODEL_COLORS dict, with fallback to palette."""
    palette = sns.color_palette(palette_name, max(len(models), 3) * 3)
    next_idx = len(color_map)
    for model in sorted(models):
        if model not in color_map:
            # Try to use predefined color first, fallback to palette
            if model in MODEL_COLORS:
                color_map[model] = MODEL_COLORS[model]
            else:
                color_map[model] = palette[next_idx % len(palette)]
                next_idx += 1


def _extract_split_seed(path: Path) -> Optional[str]:
    for part in path.parts:
        if part.startswith("split_seed_"):
            return part
    return None


def load_rocpr_summary_stats(base_dir: Path, stage: str) -> Dict[str, Dict[str, float]]:
    """Load ROC-AUC / PR-AUC mean+std for legend display.

    External stage uses the global aggregated CSV (one value per split_seed, summarized across seeds).
    CV stage intentionally summarizes across split_seeds (not across folds), matching the paper-style
    statement "mean±std across seeds".
    """
    stats: Dict[str, Dict[str, float]] = {}
    results_dir = base_dir / "results"
    if stage == "external":
        path = results_dir / "all_seed_external_summary.csv"
        if not path.exists():
            return stats
        df = pd.read_csv(path)
        needed = {"model", "roc_auc_mean", "roc_auc_std", "pr_auc_mean", "pr_auc_std"}
        if not needed.issubset(df.columns):
            return stats
        for _, row in df.iterrows():
            model = str(row["model"])
            stats[model] = {
                "roc_auc_mean": float(row["roc_auc_mean"]),
                "roc_auc_std": float(row["roc_auc_std"]),
                "pr_auc_mean": float(row["pr_auc_mean"]),
                "pr_auc_std": float(row["pr_auc_std"]),
            }
        return stats

    if stage == "cv":
        # Prefer the explicit across-seed summary produced by step11_training_summary.py if available.
        # This is the most unambiguous "across split_seed" definition and matches the user's tables.
        across_path = results_dir / "all_seed_cv_summary_across_seeds.csv"
        if across_path.exists():
            try:
                df = pd.read_csv(across_path)
            except Exception:
                df = None
            if df is not None and {"model", "metric", "mean", "std"}.issubset(df.columns):
                df["metric"] = df["metric"].astype(str).str.strip().str.lower()
                for model, group in df.groupby("model"):
                    model_stats: Dict[str, float] = {}
                    for _, row in group.iterrows():
                        metric = str(row["metric"])
                        if metric in ("roc_auc", "pr_auc"):
                            if pd.isna(row.get("mean")) or pd.isna(row.get("std")):
                                continue
                            model_stats[f"{metric}_mean"] = float(row["mean"])
                            model_stats[f"{metric}_std"] = float(row["std"])
                    if model_stats:
                        stats[str(model)] = model_stats
                return stats

        # Cross-seed summary for CV: for each split_seed, read cv_summary.csv (mean over folds),
        # then compute mean/std across split_seeds. This avoids mixing fold-to-fold variation
        # into the "across seeds" error bars shown in legends.
        per_model: Dict[str, Dict[str, List[float]]] = {}
        for split_dir in split_seed_dirs(base_dir):
            path = split_dir / "results" / "cv_summary.csv"
            if not path.exists():
                continue
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            needed = {"model", "metric", "mean"}
            if not needed.issubset(df.columns):
                continue
            df["metric"] = df["metric"].astype(str).str.strip().str.lower()
            for _, row in df.iterrows():
                metric = str(row["metric"])
                if metric not in ("roc_auc", "pr_auc"):
                    continue
                val = row.get("mean")
                if pd.isna(val):
                    continue
                model = str(row["model"])
                per_model.setdefault(model, {"roc_auc": [], "pr_auc": []})
                per_model[model][metric].append(float(val))

        for model, buckets in per_model.items():
            model_stats: Dict[str, float] = {}
            for metric in ("roc_auc", "pr_auc"):
                values = buckets.get(metric, [])
                if not values:
                    continue
                arr = np.asarray(values, dtype=float)
                model_stats[f"{metric}_mean"] = float(np.mean(arr))
                # Match seed_cv_metrics_summary.csv (pandas std default ddof=1, i.e. sample std).
                model_stats[f"{metric}_std"] = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
            if model_stats:
                stats[model] = model_stats
        return stats

    return stats


def prepare_curves(prediction_files: List[Path], stage: str) -> tuple[Dict[str, Dict[str, List[np.ndarray]]], float]:
    curves: Dict[str, Dict[str, List[np.ndarray]]] = {}
    fpr_grid = np.linspace(0.0, 1.0, 400)
    recall_grid = np.linspace(0.0, 1.0, 400)
    pooled_y_true: List[np.ndarray] = []

    def record_curve(model_name: str, y_true: np.ndarray, scores: np.ndarray, file_path: Path) -> None:
        if len(scores) == 0:
            print(f"[WARN] Empty scores for model '{model_name}' in {file_path}")
            return
        if np.unique(y_true).size < 2:
            print(f"[WARN] Single-class labels for model '{model_name}' in {file_path}; skipped.")
            return
        if np.all(np.isnan(scores)):
            print(f"[WARN] All scores are NaN for model '{model_name}' in {file_path}; skipped.")
            return
        valid = ~np.isnan(scores)
        y_true = y_true[valid]
        scores = scores[valid]
        if len(scores) == 0 or np.unique(y_true).size < 2:
            print(f"[WARN] Invalid/filtered samples for model '{model_name}' in {file_path}; skipped.")
            return
        fpr, tpr, _ = roc_curve(y_true, scores)
        precision, recall, _ = precision_recall_curve(y_true, scores)
        
        exact_roc_auc = roc_auc_score(y_true, scores)
        exact_pr_auc = average_precision_score(y_true, scores)
        
        entry = curves.setdefault(model_name, {"roc": [], "pr": [], "exact_roc_auc": [], "exact_pr_auc": []})
        entry["roc"].append(interpolate_curve(fpr, tpr, fpr_grid))
        entry.setdefault("roc_grid", fpr_grid)
        entry["pr"].append(interpolate_curve(recall[::-1], precision[::-1], recall_grid))
        entry.setdefault("pr_grid", recall_grid)
        entry["exact_roc_auc"].append(exact_roc_auc)
        entry["exact_pr_auc"].append(exact_pr_auc)
        pooled_y_true.append(y_true.astype(float))

    required = {"y_true", "y_prob", "model"}
    if stage == "cv":
        # For CV, aggregate folds within each split_seed_* first.
        # This matches the pipeline summaries (one metric value per seed), so std aligns with CSV.
        by_seed: Dict[str, List[Path]] = {}
        for path in prediction_files:
            seed_dir = _extract_split_seed(path)
            if not seed_dir:
                print(f"[WARN] Could not infer split_seed_* from {path}; skipped.")
                continue
            by_seed.setdefault(seed_dir, []).append(path)

        for seed_dir, paths in sorted(by_seed.items()):
            dfs = []
            for path in sorted(paths):
                try:
                    df = pd.read_csv(path)
                except Exception as exc:
                    print(f"[WARN] Failed to read {path}: {exc}")
                    continue
                if not required.issubset(df.columns):
                    print(f"[WARN] Missing required columns in {path}. Required: {sorted(required)}")
                    continue
                dfs.append(df)
            if not dfs:
                continue
            merged = pd.concat(dfs, ignore_index=True)
            for model_name, group in merged.groupby("model"):
                y_true = pd.to_numeric(group["y_true"], errors="coerce").to_numpy()
                scores = pd.to_numeric(group["y_prob"], errors="coerce").to_numpy()
                if np.all(np.isnan(y_true)):
                    print(f"[WARN] y_true is empty/NaN for model '{model_name}' in {seed_dir}")
                    continue
                record_curve(str(model_name), y_true, scores, Path(seed_dir))
    else:
        for path in prediction_files:
            try:
                df = pd.read_csv(path)
            except Exception as exc:
                print(f"[WARN] Failed to read {path}: {exc}")
                continue
            if not required.issubset(df.columns):
                print(f"[WARN] Missing required columns in {path}. Required: {sorted(required)}")
                continue
            for model_name, group in df.groupby("model"):
                y_true = pd.to_numeric(group["y_true"], errors="coerce").to_numpy()
                scores = pd.to_numeric(group["y_prob"], errors="coerce").to_numpy()
                if np.all(np.isnan(y_true)):
                    print(f"[WARN] y_true is empty/NaN for model '{model_name}' in {path}")
                    continue
                record_curve(str(model_name), y_true, scores, path)
    prevalence = float(np.mean(np.concatenate(pooled_y_true))) if pooled_y_true else 0.0
    return curves, prevalence


# %%
def plot_roc_pr(curves: Dict[str, Dict[str, List[np.ndarray]]],
                prevalence: float,
                output_path: Path,
                stage: str,
                palette_name: str,
                dpi: int,
                font: str,
                global_color_map: Dict[str, tuple],
                error_type: str = "ci95",
                summary_stats: Optional[Dict[str, Dict[str, float]]] = None) -> None:
    if not curves:
        return
    configure_plotting(font)
    
    import matplotlib.ticker as ticker
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    model_entries: List[Dict[str, object]] = []
    for model, data in curves.items():
        if not data["roc"]:
            continue
        fpr_grid = data["roc_grid"]
        recall_grid = data["pr_grid"]
        mean_tpr = np.mean(data["roc"], axis=0)
        err_tpr = error_from_samples(data["roc"], error_type)
        mean_prec = np.mean(data["pr"], axis=0)
        err_prec = error_from_samples(data["pr"], error_type)
        roc_auc_vals = data.get("exact_roc_auc", [auc(fpr_grid, arr) for arr in data["roc"]])
        pr_auc_vals = data.get("exact_pr_auc", [auc(recall_grid, arr) for arr in data["pr"]])
        # If requested, force legend mean/std to match pipeline CSV exactly.
        if summary_stats and error_type == "std":
            model_stat = summary_stats.get(str(model))
        else:
            model_stat = None
        model_entries.append({
            "model": model,
            "fpr_grid": fpr_grid,
            "recall_grid": recall_grid,
            "mean_tpr": mean_tpr,
            "err_tpr": err_tpr,
            "mean_prec": mean_prec,
            "err_prec": err_prec,
            "roc_auc_mean": float(model_stat["roc_auc_mean"]) if model_stat and "roc_auc_mean" in model_stat else float(np.mean(roc_auc_vals)),
            "roc_auc_err": float(model_stat["roc_auc_std"]) if model_stat and "roc_auc_std" in model_stat else error_from_scalars(roc_auc_vals, error_type),
            "pr_auc_mean": float(model_stat["pr_auc_mean"]) if model_stat and "pr_auc_mean" in model_stat else float(np.mean(pr_auc_vals)),
            "pr_auc_err": float(model_stat["pr_auc_std"]) if model_stat and "pr_auc_std" in model_stat else error_from_scalars(pr_auc_vals, error_type),
        })

    roc_sorted = sorted(
        model_entries,
        key=lambda d: (d["roc_auc_mean"], d["roc_auc_err"]),
        reverse=True,
    )
    pr_sorted = sorted(
        model_entries,
        key=lambda d: (d["pr_auc_mean"], d["pr_auc_err"]),
        reverse=True,
    )
    get_or_assign_model_colors([str(entry["model"]) for entry in model_entries], palette_name, global_color_map)

    for entry in roc_sorted:
        model = str(entry["model"])
        color = global_color_map[model]
        fpr_grid = np.asarray(entry["fpr_grid"])
        mean_tpr = np.asarray(entry["mean_tpr"])
        err_tpr = np.asarray(entry["err_tpr"])
        axes[0].plot(
            fpr_grid,
            mean_tpr,
            label=(f"{model} (ROC-AUC={entry['roc_auc_mean']:.{LEGEND_DECIMALS}f}"
                   f"±{entry['roc_auc_err']:.{LEGEND_DECIMALS}f})"),
            linewidth=1.5,
            color=color,
        )
        axes[0].fill_between(
            fpr_grid,
            np.clip(mean_tpr - err_tpr, 0, 1),
            np.clip(mean_tpr + err_tpr, 0, 1),
            color=color,
            alpha=0.1,
            linewidth=0,
        )

    for entry in pr_sorted:
        model = str(entry["model"])
        color = global_color_map[model]
        recall_grid = np.asarray(entry["recall_grid"])
        mean_prec = np.asarray(entry["mean_prec"])
        err_prec = np.asarray(entry["err_prec"])
        axes[1].plot(
            recall_grid,
            mean_prec,
            label=(f"{model} (PR-AUC={entry['pr_auc_mean']:.{LEGEND_DECIMALS}f}"
                   f"±{entry['pr_auc_err']:.{LEGEND_DECIMALS}f})"),
            linewidth=1.5,
            color=color,
        )
        axes[1].fill_between(
            recall_grid,
            np.clip(mean_prec - err_prec, 0, 1),
            np.clip(mean_prec + err_prec, 0, 1),
            color=color,
            alpha=0.1,
            linewidth=0,
        )
    axes[0].plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, color="gray", label="Random Baseline")
    axes[1].axhline(
        prevalence,
        linestyle="--",
        linewidth=1.0,
        color="gray",
        label=f"Positive-Rate Baseline ({prevalence:.{LEGEND_DECIMALS}f})",
    )
    stage_titles = {
        "cv": "Cross-Validation",
        "external": "External Test Set",
    }
    stage_label = stage_titles.get(stage.lower(), stage.replace("_", " ").title())
    axes[0].set_title(f"(A) {stage_label} ROC Curve")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(
        loc="lower right",
        frameon=True,
        fancybox=False,
        edgecolor="black",
        facecolor="white",
        framealpha=1.0,
    )
    axes[1].set_title(f"(B) {stage_label} PR Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(
        loc="lower right",
        frameon=True,
        fancybox=False,
        edgecolor="black",
        facecolor="white",
        framealpha=1.0,
    )
    
    for ax in axes:
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)
        
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda y, pos: "" if y == 0.0 else f"{y:.1f}")
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg", dpi=dpi)
    plt.close(fig)


# %%
def normalize_metric_names(metric_text: str) -> List[str]:
    metrics: List[str] = []
    for token in metric_text.split(","):
        key = token.strip().lower().replace("%", "").replace("-", "_")
        key = key.replace("prauc", "pr_auc").replace("rocauc", "roc_auc")
        if not key:
            continue
        mapped = METRIC_ALIASES.get(key)
        if mapped and mapped not in metrics:
            metrics.append(mapped)
    return metrics or DEFAULT_METRICS


def filter_boxplot_metrics(metrics: List[str]) -> List[str]:
    filtered = [m for m in metrics if m in BOXPLOT_ALLOWED_METRICS]
    dropped = [m for m in metrics if m not in BOXPLOT_ALLOWED_METRICS]
    if dropped:
        print(f"[INFO] Dropped non-boxplot metrics: {', '.join(dropped)}")
    return filtered or DEFAULT_METRICS


def parse_seed_from_name(split_dir: Path) -> Optional[int]:
    try:
        return int(split_dir.name.split("_")[-1])
    except ValueError:
        return None


# %%
def prepare_metric_dataframe(base_dir: Path, stage: str, metrics: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for split_dir in split_seed_dirs(base_dir):
        seed = parse_seed_from_name(split_dir)
        if stage == "external":
            path = split_dir / "results" / "external_test_results.csv"
            if not path.exists():
                print(f"[WARN] Missing file: {path}")
                continue
            try:
                df = pd.read_csv(path)
            except Exception as exc:
                print(f"[WARN] Failed to read {path}: {exc}")
                continue
            if "model" not in df.columns:
                print(f"[WARN] Missing 'model' column in {path}")
                continue
            for _, row in df.iterrows():
                for metric in metrics:
                    if metric not in df.columns:
                        continue
                    value = row.get(metric)
                    if pd.isna(value):
                        continue
                    rows.append({
                        "stage": stage,
                        "metric": metric,
                        "metric_label": METRIC_LABELS.get(metric, metric),
                        "model": row["model"],
                        "split_seed": seed,
                        "value": float(value),
                    })
        elif stage == "cv":
            path = split_dir / "results" / "cv_summary.csv"
            if not path.exists():
                print(f"[WARN] Missing file: {path}")
                continue
            try:
                df = pd.read_csv(path)
            except Exception as exc:
                print(f"[WARN] Failed to read {path}: {exc}")
                continue
            needed = {"model", "metric", "mean"}
            if not needed.issubset(df.columns):
                print(f"[WARN] Missing required columns in {path}. Required: {sorted(needed)}")
                continue
            for _, row in df.iterrows():
                metric = str(row["metric"]).strip().lower()
                if metric not in metrics:
                    continue
                value = row.get("mean")
                if pd.isna(value):
                    continue
                rows.append({
                    "stage": stage,
                    "metric": metric,
                    "metric_label": METRIC_LABELS.get(metric, metric),
                    "model": row["model"],
                    "split_seed": seed,
                    "value": float(value),
                })
    return pd.DataFrame(rows)


# %%
def plot_metric_boxplots(metric_df: pd.DataFrame,
                         metrics: List[str],
                         stage: str,
                         output_path: Path,
                         palette_name: str,
                         dpi: int,
                         font: str,
                         global_color_map: Dict[str, tuple]) -> None:
    if metric_df.empty:
        return
    configure_plotting(font)
    n_metrics = len(metrics)
    n_cols = 2
    n_rows = int(np.ceil(n_metrics / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9 * n_cols, 4.5 * n_rows), sharey=False)
    axes_arr = np.array(axes).reshape(-1)
    model_order = sorted(metric_df["model"].unique())
    get_or_assign_model_colors(model_order, palette_name, global_color_map)
    box_palette = {model: global_color_map[model] for model in model_order}
    for idx, metric in enumerate(metrics):
        ax = axes_arr[idx]
        sub_df = metric_df[metric_df["metric"] == metric]
        if sub_df.empty:
            ax.set_visible(False)
            continue
        sns.boxplot(
            x="model",
            y="value",
            data=sub_df,
            ax=ax,
            order=model_order,
            palette=box_palette,
            showcaps=True,
            boxprops={"alpha": 0.7},
            showfliers=False,
        )
        sns.swarmplot(
            x="model",
            y="value",
            data=sub_df,
            ax=ax,
            order=model_order,
            color="black",
            size=3,
            alpha=0.45,
        )
        ax.set_title(f"{stage.capitalize()} {METRIC_LABELS.get(metric, metric)}")
        ax.set_xlabel("Model")
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.tick_params(axis="x", rotation=45)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)
    for idx in range(n_metrics, len(axes_arr)):
        axes_arr[idx].set_visible(False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg", dpi=dpi)
    plt.close(fig)


# %%
def plot_polar_metric_bars(
    metric_df: pd.DataFrame,
    stage: str,
    output_path: Path,
    palette_name: str,
    dpi: int,
    font: str,
    global_color_map: Dict[str, tuple],
    show_values: bool = True,
    error_type: str = "ci95",
) -> None:
    if metric_df.empty:
        return
    configure_plotting(font)

    models = sorted(metric_df["model"].dropna().unique())
    if not models:
        print(f"[WARN] No models available for polar plot at stage '{stage}'.")
        return

    get_or_assign_model_colors(models, palette_name, global_color_map)
    fig, ax = plt.subplots(1, 1, figsize=(12, 12), subplot_kw={"projection": "polar"})

    n_metrics = len(POLAR_METRIC_ORDER)
    n_models = len(models)

    sector_width = 2 * np.pi / n_metrics
    total_bars = n_metrics * n_models
    group_gap_units = 1.5
    unit_angle = 2 * np.pi / (total_bars + n_metrics * group_gap_units)
    bar_slot = unit_angle
    group_gap = group_gap_units * unit_angle
    bar_width = bar_slot * 0.92
    inner_radius = 0.25
    radial_scale = 0.62

    theta_centers: List[float] = []
    group_centers: List[float] = []
    theta_cursor = 0.0
    for _ in POLAR_METRIC_ORDER:
        bar_centers = [theta_cursor + (i + 0.5) * bar_slot for i in range(n_models)]
        theta_centers.extend(bar_centers)
        group_centers.append(float(np.mean(bar_centers)))
        theta_cursor += n_models * bar_slot + group_gap

    # Re-center so MCC group center sits exactly at 12 o'clock.
    center_shift = -group_centers[0]
    theta_centers = [((t + center_shift) % (2 * np.pi)) for t in theta_centers]
    group_centers = [((t + center_shift) % (2 * np.pi)) for t in group_centers]

    bar_idx = 0
    for metric in POLAR_METRIC_ORDER:
        metric_df_sub = metric_df[metric_df["metric"] == metric]
        for model in models:
            vals = pd.to_numeric(
                metric_df_sub.loc[metric_df_sub["model"] == model, "value"],
                errors="coerce",
            ).dropna().to_numpy()
            if vals.size == 0:
                print(f"[WARN] Missing values for metric '{metric}' and model '{model}' at stage '{stage}'.")
                bar_idx += 1
                continue
            mean_val = float(np.mean(vals))
            err_val = error_from_scalars(vals.tolist(), error_type)
            theta = theta_centers[bar_idx]
            bar_idx += 1

            #height = np.clip(mean_val, 0.0, 1.0) * radial_scale
            height = np.clip(mean_val, 0.0, 1.0) * radial_scale
            y_top = inner_radius + height
            color = global_color_map.get(model, (0.5, 0.5, 0.5))

            ci_low = max(0.0, mean_val - err_val)
            ci_high = min(1.0, mean_val + err_val)
            err_low = (mean_val - ci_low) * radial_scale
            err_high = (ci_high - mean_val) * radial_scale
            color = global_color_map.get(model, sns.color_palette(palette_name)[0])

            ax.bar(
                theta,
                height,
                bottom=inner_radius,
                width=bar_width,
                color=color,
                alpha=0.90,
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
            ax.errorbar(
                theta,
                y_top,
                yerr=np.array([[err_low], [err_high]]),
                fmt="none",
                ecolor="black",
                elinewidth=0.8,
                capsize=1.6,
                capthick=0.8,
                zorder=4,
            )
            if show_values:
                # Place value text inside bar; MCC at 0.4 ring, others at 0.6 ring.
                target_ring = 0.4 if metric == "mcc" else 0.6
                target_r = inner_radius + target_ring * radial_scale
                label_r = min(target_r, y_top - 0.02)
                label_r = max(label_r, inner_radius + 0.02)
                text_rotation = np.degrees(theta) - 90.0
                if metric in {"precision", "recall"}:
                    text_rotation += 180.0
                elif metric == "mcc" and theta > np.pi:
                    text_rotation += 180.0
                ax.text(
                    theta,
                    label_r,
                    f"{mean_val:.3f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color="black",
                    rotation=text_rotation,
                    rotation_mode="anchor",
                    zorder=5,
                )

    ax.set_theta_offset(np.pi / 2.0)  # MCC sector around 12 o'clock
    ax.set_theta_direction(-1)
    ax.set_xticks([])
    ax.set_ylim(0.0, 1.1)
    ax.set_axis_off() # 移除默认轴

    # 添加自定义背景圈
    radial_label_theta = np.deg2rad(36.0)  # 36° clockwise from top (with theta_direction=-1)
    for r in [0.2, 0.4, 0.6, 0.8, 1.0]:
        r_pos = inner_radius + r * radial_scale
        ax.plot(np.linspace(0, 2*np.pi, 100), [r_pos]*100, color="gray", alpha=0.2, ls="--", lw=0.8)
        ax.text(
            radial_label_theta,
            r_pos - 0.015,
            f"{r}",
            color="black",
            fontsize=8,
            ha="center",
            va="center",
        )

    # 绘制指标标签 (在中心区域)
    for metric, theta in zip(POLAR_METRIC_ORDER, group_centers):
        label_text = METRIC_LABELS.get(metric, metric.upper())
        ax.text(theta, inner_radius - 0.07, label_text, ha="center", va="center",
                fontsize=13, fontweight="bold")
        
    tick_values = [0.2, 0.4, 0.6, 0.8, 1.0]
    tick_positions = [inner_radius + v * radial_scale for v in tick_values]
    ax.set_yticks(tick_positions)
    ax.set_yticklabels([f"{v:.1f}" for v in tick_values], fontweight="semibold")
    ax.set_rlabel_position(36) ## Move radial labels slightly away from the first metric sector
    ax.grid(True, axis="y", alpha=0.22, linewidth=0.8, color="#777777")
    ax.grid(False, axis="x")
    # Emphasize the outer (1.0) ring in black.
    theta_full = np.linspace(0, 2 * np.pi, 720)
    ax.plot(theta_full, np.full_like(theta_full, tick_positions[-1]), color="black", linewidth=1.2, zorder=2)
    ax.set_facecolor("white")
    ax.spines["polar"].set_visible(False)

    # Inner ring with metric labels.
    inner_circle = Circle((0.5, 0.5), 0.120, transform=ax.transAxes, facecolor="white", edgecolor="black", linewidth=1.0, zorder=10)
    ax.add_artist(inner_circle)
    for metric, theta in zip(POLAR_METRIC_ORDER, group_centers):
        label_text = METRIC_LABELS.get(metric, metric.upper())
        ax.text(
            theta,
            inner_radius * 0.5,
            label_text,
            ha="center",
            va="center",
            fontsize=12.5,
            fontweight="semibold",
            color="black",
            rotation=0,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.18, "alpha": 1.0},
            zorder=12,
        )

    stage_titles = {
        "cv": "Cross-Validation",
        "external": "External Test Set",
    }
    stage_label = stage_titles.get(stage.lower(), stage.replace("_", " ").title())
    ax.set_title(f"{stage_label} Grouped Circular Bar Chart", pad=8, fontsize=14)
    legend_handles = [
        Patch(facecolor=global_color_map[model], edgecolor="black", linewidth=0.8, label=str(model))
        for model in models
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.50, 0.05),
        bbox_transform=fig.transFigure,
        ncol=len(models),
        frameon=True,
        fancybox=False,
        edgecolor="black",
        facecolor="white",
        framealpha=1.0,
    )
    fig.tight_layout(rect=[0.03, 0.05, 0.95, 0.99])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg", dpi=dpi)
    plt.close(fig)


# %%
def main() -> None:
    args = parse_args()
    base_dir = resolve_base_dir(args.base_dir)
    output_dir = args.output_dir or (base_dir / "figures")
    include_external = args.include_external
    include_cv = args.include_cv
    if not include_external and not include_cv:
        include_external = True
        include_cv = True
    metrics = filter_boxplot_metrics(normalize_metric_names(args.boxplot_metrics))

    stages: List[str] = []
    if include_external:
        stages.append("external")
    if include_cv:
        stages.append("cv")

    print(f"[INFO] Base run directory: {base_dir}")
    print(f"[INFO] Figures will be saved to: {output_dir}")

    global_color_map: Dict[str, tuple] = {}
    for stage in stages:
        prediction_files = collect_prediction_files(base_dir, stage)
        if not prediction_files:
            print(f"[WARN] No prediction files found for stage '{stage}'.")
            continue
        curves, prevalence = prepare_curves(prediction_files, stage)
        if not curves:
            print(f"[WARN] Could not build curves for stage '{stage}'.")
            continue
        summary_stats = load_rocpr_summary_stats(base_dir, stage)
        rocpr_path = output_dir / f"{stage}_roc_pr.svg"
        plot_roc_pr(
            curves,
            prevalence,
            rocpr_path,
            stage,
            args.palette,
            args.dpi,
            args.font,
            global_color_map,
            args.error_type,
            summary_stats=summary_stats,
        )
        print(f"[OK] Saved {stage} ROC/PR: {rocpr_path}")

    boxplot_stages: List[str] = []
    if args.boxplot_stage in ("external", "both"):
        boxplot_stages.append("external")
    if args.boxplot_stage in ("cv", "both"):
        boxplot_stages.append("cv")

    for stage in boxplot_stages:
        metric_df = prepare_metric_dataframe(base_dir, stage, metrics)
        if metric_df.empty:
            print(f"[WARN] No metric data found for stage '{stage}'.")
            continue
        boxplot_path = output_dir / f"{stage}_metric_boxplots.svg"
        plot_metric_boxplots(metric_df, metrics, stage, boxplot_path, args.palette, args.dpi, args.font, global_color_map)
        print(f"[OK] Saved {stage} metric boxplots: {boxplot_path}")
        polar_path = output_dir / f"{stage}_polar_metric_bars.svg"
        plot_polar_metric_bars(
            metric_df,
            stage,
            polar_path,
            args.palette,
            args.dpi,
            args.font,
            global_color_map,
            show_values=args.polar_show_values,
            error_type=args.error_type,
        )
        print(f"[OK] Saved {stage} polar metric bars: {polar_path}")


# %%
if __name__ == "__main__":
    main()
