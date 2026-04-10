"""Plot QSAR performance figures from step10_qsar_ml.py outputs.

Usage examples:
    python scripts/step40_plot_performance.py
    python scripts/step40_plot_performance.py --base-dir models_out/qsar_ml_20260409_122600
    python scripts/step40_plot_performance.py --include-external --include-cv --boxplot-stage both

python scripts/step40_plot_performance.py \
  --base-dir models_out/qsar_ml_20260409_214751 \
  --include-external \
  --include-cv \
  --boxplot-stage both

  
optional arguments:
--boxplot-metrics mcc,f1,accuracy,roc_auc,pr_auc
--palette colorblind
--dpi 600
--font Cambria
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
from sklearn.metrics import auc, precision_recall_curve, roc_curve


DEFAULT_METRICS = ["mcc", "f1", "accuracy", "roc_auc", "pr_auc", "ef1", "ef5", "ef10", "nef10"]
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
    "ef1": "ef1",
    "ef5": "ef5",
    "ef10": "ef10",
    "nef10": "nef10",
    "nef": "nef10",
}
METRIC_LABELS = {
    "accuracy": "ACC",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "mcc": "MCC",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
    "ef1": "EF1",
    "ef5": "EF5",
    "ef10": "EF10",
    "nef10": "NEF10",
}


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
        help="Comma-separated metrics (aliases allowed: ACC/AUC/PR_AUC etc.; e.g. mcc,f1,accuracy,roc_auc,pr_auc,ef1,ef5,ef10,nef10)",
    )
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


def interpolate_curve(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.interp(grid, x, y, left=y[0], right=y[-1])


def prepare_curves(prediction_files: List[Path]) -> Dict[str, Dict[str, List[np.ndarray]]]:
    curves: Dict[str, Dict[str, List[np.ndarray]]] = {}
    fpr_grid = np.linspace(0.0, 1.0, 400)
    recall_grid = np.linspace(0.0, 1.0, 400)

    def record_curve(model_name: str, y_true: np.ndarray, scores: np.ndarray) -> None:
        if len(scores) == 0:
            return
        if np.unique(y_true).size < 2:
            return
        if np.all(np.isnan(scores)):
            return
        valid = ~np.isnan(scores)
        y_true = y_true[valid]
        scores = scores[valid]
        if len(scores) == 0 or np.unique(y_true).size < 2:
            return
        fpr, tpr, _ = roc_curve(y_true, scores)
        precision, recall, _ = precision_recall_curve(y_true, scores)
        entry = curves.setdefault(model_name, {"roc": [], "pr": []})
        entry["roc"].append(interpolate_curve(fpr, tpr, fpr_grid))
        entry.setdefault("roc_grid", fpr_grid)
        entry["pr"].append(interpolate_curve(recall[::-1], precision[::-1], recall_grid))
        entry.setdefault("pr_grid", recall_grid)

    for path in prediction_files:
        df = pd.read_csv(path)
        required = {"y_true", "y_prob", "model"}
        if not required.issubset(df.columns):
            continue
        for model_name, group in df.groupby("model"):
            y_true = group["y_true"].astype(float).to_numpy()
            scores = group["y_prob"].astype(float).to_numpy()
            record_curve(str(model_name), y_true, scores)
    return curves


def plot_roc_pr(curves: Dict[str, Dict[str, List[np.ndarray]]],
                output_path: Path,
                stage: str,
                palette_name: str,
                dpi: int,
                font: str) -> None:
    if not curves:
        return
    configure_plotting(font)
    colors = sns.color_palette(palette_name) * 3
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    model_entries: List[Dict[str, object]] = []
    for model, data in curves.items():
        if not data["roc"]:
            continue
        fpr_grid = data["roc_grid"]
        recall_grid = data["pr_grid"]
        mean_tpr = np.mean(data["roc"], axis=0)
        std_tpr = np.std(data["roc"], axis=0)
        mean_prec = np.mean(data["pr"], axis=0)
        std_prec = np.std(data["pr"], axis=0)
        roc_auc_vals = [auc(fpr_grid, arr) for arr in data["roc"]]
        pr_auc_vals = [auc(recall_grid, arr) for arr in data["pr"]]
        model_entries.append({
            "model": model,
            "fpr_grid": fpr_grid,
            "recall_grid": recall_grid,
            "mean_tpr": mean_tpr,
            "std_tpr": std_tpr,
            "mean_prec": mean_prec,
            "std_prec": std_prec,
            "roc_auc_mean": float(np.mean(roc_auc_vals)),
            "roc_auc_std": float(np.std(roc_auc_vals)),
            "pr_auc_mean": float(np.mean(pr_auc_vals)),
            "pr_auc_std": float(np.std(pr_auc_vals)),
        })

    roc_sorted = sorted(
        model_entries,
        key=lambda d: (d["roc_auc_mean"], d["roc_auc_std"]),
        reverse=True,
    )
    pr_sorted = sorted(
        model_entries,
        key=lambda d: (d["pr_auc_mean"], d["pr_auc_std"]),
        reverse=True,
    )

    color_map = {entry["model"]: colors[idx % len(colors)] for idx, entry in enumerate(roc_sorted)}

    for entry in roc_sorted:
        model = str(entry["model"])
        color = color_map[model]
        fpr_grid = np.asarray(entry["fpr_grid"])
        mean_tpr = np.asarray(entry["mean_tpr"])
        std_tpr = np.asarray(entry["std_tpr"])
        axes[0].plot(
            fpr_grid,
            mean_tpr,
            label=f"{model} (ROC-AUC={entry['roc_auc_mean']:.3f}±{entry['roc_auc_std']:.3f})",
            linewidth=1.5,
            color=color,
        )
        axes[0].fill_between(
            fpr_grid,
            np.clip(mean_tpr - std_tpr, 0, 1),
            np.clip(mean_tpr + std_tpr, 0, 1),
            color=color,
            alpha=0.1,
            linewidth=0,
        )

    for entry in pr_sorted:
        model = str(entry["model"])
        color = color_map.get(model, colors[0])
        recall_grid = np.asarray(entry["recall_grid"])
        mean_prec = np.asarray(entry["mean_prec"])
        std_prec = np.asarray(entry["std_prec"])
        axes[1].plot(
            recall_grid,
            mean_prec,
            label=f"{model} (PR-AUC={entry['pr_auc_mean']:.3f}±{entry['pr_auc_std']:.3f})",
            linewidth=1.5,
            color=color,
        )
        axes[1].fill_between(
            recall_grid,
            np.clip(mean_prec - std_prec, 0, 1),
            np.clip(mean_prec + std_prec, 0, 1),
            color=color,
            alpha=0.1,
            linewidth=0,
        )
    axes[0].set_title(f"{stage.capitalize()} ROC Curve")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(loc="lower right", frameon=False)
    axes[1].set_title(f"{stage.capitalize()} PR Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(loc="lower left", frameon=False)
    for ax in axes:
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    sns.despine(fig=fig, left=False, bottom=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg", dpi=dpi)
    plt.close(fig)


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


def parse_seed_from_name(split_dir: Path) -> Optional[int]:
    try:
        return int(split_dir.name.split("_")[-1])
    except ValueError:
        return None


def prepare_metric_dataframe(base_dir: Path, stage: str, metrics: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for split_dir in split_seed_dirs(base_dir):
        seed = parse_seed_from_name(split_dir)
        if stage == "external":
            path = split_dir / "results" / "external_test_results.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
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
                continue
            df = pd.read_csv(path)
            needed = {"model", "metric", "mean"}
            if not needed.issubset(df.columns):
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


def plot_metric_boxplots(metric_df: pd.DataFrame,
                         metrics: List[str],
                         stage: str,
                         output_path: Path,
                         palette_name: str,
                         dpi: int,
                         font: str) -> None:
    if metric_df.empty:
        return
    configure_plotting(font)
    n_metrics = len(metrics)
    n_cols = 2
    n_rows = int(np.ceil(n_metrics / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9 * n_cols, 4.5 * n_rows), sharey=False)
    axes_arr = np.array(axes).reshape(-1)
    model_order = sorted(metric_df["model"].unique())
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
            palette=palette_name,
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
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for idx in range(n_metrics, len(axes_arr)):
        axes_arr[idx].set_visible(False)
    sns.despine(fig=fig, left=True, bottom=True)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg", dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    base_dir = resolve_base_dir(args.base_dir)
    output_dir = args.output_dir or (base_dir / "figures")
    include_external = args.include_external
    include_cv = args.include_cv
    if not include_external and not include_cv:
        include_external = True
        include_cv = True
    metrics = normalize_metric_names(args.boxplot_metrics)

    stages: List[str] = []
    if include_external:
        stages.append("external")
    if include_cv:
        stages.append("cv")

    print(f"[INFO] Base run directory: {base_dir}")
    print(f"[INFO] Figures will be saved to: {output_dir}")

    for stage in stages:
        prediction_files = collect_prediction_files(base_dir, stage)
        if not prediction_files:
            print(f"[WARN] No prediction files found for stage '{stage}'.")
            continue
        curves = prepare_curves(prediction_files)
        if not curves:
            print(f"[WARN] Could not build curves for stage '{stage}'.")
            continue
        rocpr_path = output_dir / f"{stage}_roc_pr.svg"
        plot_roc_pr(curves, rocpr_path, stage, args.palette, args.dpi, args.font)
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
        plot_metric_boxplots(metric_df, metrics, stage, boxplot_path, args.palette, args.dpi, args.font)
        print(f"[OK] Saved {stage} metric boxplots: {boxplot_path}")


if __name__ == "__main__":
    main()
