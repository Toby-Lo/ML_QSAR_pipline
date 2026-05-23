"""Enhanced QSAR performance visualization with multiple plot styles.

Features:
- 5 complementary visualization approaches (A, B, C, D, E)
- Consistent model color mapping across all plots
- Support for both CV and External Test data
- Error bar options: Std or CI95 (95% confidence interval)
- Publication-ready SVG and optional TIFF output at 600 DPI

Usage examples:
    python scripts/step40_plot_performance_enhanced.py
    python scripts/step40_plot_performance_enhanced.py \\
        --base-dir models_out/qsar_ml_20260412_162829 \\
        --output-dir models_out/qsar_ml_20260412_162829/figures/enhanced/ \\
        --include-external --include-cv \\
        --error-type ci95
    
    python scripts/step40_plot_performance_enhanced.py \\
        --base-dir models_out/qsar_ml_20260412_162829 \\
        --output-dir models_out/qsar_ml_20260412_162829/figures/enhanced/ \\
        --stages external cv \\
        --error-type std \\
        --export-tiff

python scripts/step40_plot_performance_enhanced.py \
  --base-dir models_out/qsar_ml_20260412_162829 \
  --output-dir models_out/qsar_ml_20260412_162829/figures/performance/enhanced/ \
  --include-external --include-cv \
  --error-type std \
  --plot-types A B C D E

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams

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
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-score",
    "mcc": "MCC",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
}

CI_Z = 1.96
LEGEND_DECIMALS = 3
MODEL_COLORS = {
    'ETC': '#B08B86',   # Morandi pink/light brown
    'RFC': '#C3B083',   # Dark gold/Morandi light yellow
    'SVC': '#1F77B4',   # Core highlight blue
    'MLP': '#8F93B5',   # Morandi purple-gray/blue-gray
    'XGBC': '#8AA68A',  # Morandi bean green/gray-green
    'LR': '#A8829A'     # Morandi dark purple/pink-gray
}

# %%
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enhanced QSAR metrics visualization with multiple plot styles"
    )
    parser.add_argument(
        "--base-dir", 
        type=Path, 
        help="Run directory (e.g., models_out/qsar_ml_YYYYMMDD_HHMMSS)"
    )
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        help="Figure output directory (default: <base-dir>/figures/enhanced)"
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["external", "cv"],
        default=["external", "cv"],
        help="Stages to plot (default: external cv)"
    )
    parser.add_argument(
        "--error-type",
        choices=["std", "ci95"],
        default="std",
        help="Error bar type: std (standard deviation) or ci95 (95%% confidence interval)"
    )
    parser.add_argument(
        "--include-external",
        action="store_true",
        help="Include external test stage"
    )
    parser.add_argument(
        "--include-cv",
        action="store_true",
        help="Include cross-validation stage"
    )
    parser.add_argument(
        "--palette",
        default="colorblind",
        help="Seaborn palette name"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Figure DPI (default: 600)"
    )
    parser.add_argument(
        "--font",
        default="Cambria",
        help="Font for publication style (default: Cambria)"
    )
    parser.add_argument(
        "--export-tiff",
        action="store_true",
        help="Also export as TIFF (default: SVG only)"
    )
    parser.add_argument(
        "--plot-types",
        nargs="+",
        choices=["A", "B", "C", "D", "E"],
        default=["A", "B", "C", "D", "E"],
        help="Which plot types to generate (default: A B C D E)"
    )
    return parser.parse_args()


# %%
def configure_plotting(font: str) -> None:
    """Configure matplotlib for publication-quality plots."""
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = [font, "Cambria", "Times New Roman", "DejaVu Serif"]
    rcParams["text.usetex"] = False
    rcParams["font.size"] = 12
    rcParams["axes.titlesize"] = 13
    rcParams["axes.labelsize"] = 12
    rcParams["xtick.labelsize"] = 10
    rcParams["ytick.labelsize"] = 11
    rcParams["legend.fontsize"] = 10
    rcParams["axes.linewidth"] = 1.2
    rcParams["xtick.major.width"] = 1.2
    rcParams["ytick.major.width"] = 1.2


def resolve_base_dir(base_dir: Optional[Path]) -> Path:
    if base_dir and base_dir.exists():
        return base_dir.resolve()
    
    models_dir = Path("models_out")
    if not models_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir or models_dir}")
    
    subdirs = sorted([d for d in models_dir.iterdir() if d.is_dir()])
    if not subdirs:
        raise FileNotFoundError("No model directories found in models_out/")
    
    return subdirs[-1]


def split_seed_dirs(base_dir: Path) -> List[Path]:
    pattern = "split_seed_*"
    dirs = sorted(
        base_dir.glob(pattern),
        key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else 0
    )
    return dirs


def get_or_assign_model_colors(
    models: List[str], 
    palette_name: str, 
    color_map: Dict[str, tuple]
) -> None:
    palette = sns.color_palette(palette_name, max(len(models), 3) * 3)
    next_idx = len(color_map)
    for model in sorted(models):
        if model not in color_map:
            if model in MODEL_COLORS:
                color_map[model] = MODEL_COLORS[model]
            else:
                color_map[model] = palette[next_idx % len(palette)]
                next_idx += 1


# %%
def parse_seed_from_name(split_dir: Path) -> Optional[int]:
    try:
        return int(split_dir.name.split("_")[-1])
    except ValueError:
        return None


def ci95_from_scalars(values: List[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size <= 1:
        return 0.0
    sem = np.std(arr, ddof=1) / np.sqrt(arr.size)
    return float(CI_Z * sem)


def prepare_metric_dataframe(base_dir: Path, stage: str, metrics: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    
    for split_dir in split_seed_dirs(base_dir):
        seed = parse_seed_from_name(split_dir)
        
        if stage == "external":
            path = split_dir / "results" / "external_test_results.csv"
            if not path.exists(): continue
            try:
                df = pd.read_csv(path)
            except Exception: continue
            
            if "model" not in df.columns: continue
            
            for _, row in df.iterrows():
                for metric in metrics:
                    if metric not in df.columns: continue
                    value = row.get(metric)
                    if pd.isna(value): continue
                    rows.append({
                        "stage": stage,
                        "metric": metric,
                        "model": row["model"],
                        "split_seed": seed,
                        "value": float(value),
                    })
        
        elif stage == "cv":
            path = split_dir / "results" / "cv_summary.csv"
            if not path.exists(): continue
            try:
                df = pd.read_csv(path)
            except Exception: continue
            
            needed = {"model", "metric", "mean"}
            if not needed.issubset(df.columns): continue
            
            for _, row in df.iterrows():
                metric = str(row["metric"]).strip().lower()
                if metric not in metrics: continue
                value = row.get("mean")
                if pd.isna(value): continue
                rows.append({
                    "stage": stage,
                    "metric": metric,
                    "model": row["model"],
                    "split_seed": seed,
                    "value": float(value),
                })
    
    return pd.DataFrame(rows)


def aggregate_metrics_by_model(
    base_dir: Path,
    stage: str,
    metric_df: pd.DataFrame,
    metric_name: str,
    error_type: str = "std"
) -> pd.DataFrame:
    sub_df = metric_df[metric_df["metric"] == metric_name].copy()
    if sub_df.empty: return pd.DataFrame()
    
    # Use explicit summary CSVs when available to keep plotted mean/std identical to tables.
    global_df = None
    global_mode = None
    if error_type == "std":
        if stage == "cv":
            cv_path = base_dir / "results" / "all_seed_cv_summary_across_seeds.csv"
            if cv_path.exists():
                global_df = pd.read_csv(cv_path)
                global_mode = "cv_across_seeds"
        else:
            ext_path = base_dir / "results" / "all_seed_external_summary.csv"
            if ext_path.exists():
                global_df = pd.read_csv(ext_path)
                global_mode = "external_global"

    agg_data = []
    for model in sorted(sub_df["model"].unique()):
        values = sub_df[sub_df["model"] == model]["value"].to_numpy()
        
        if global_df is not None and global_mode == "external_global":
            row = global_df[global_df["model"] == model]
            if not row.empty and f"{metric_name}_mean" in row.columns and f"{metric_name}_std" in row.columns:
                mean_val = float(row.iloc[0][f"{metric_name}_mean"])
                error_val = float(row.iloc[0][f"{metric_name}_std"])
            else:
                mean_val = float(np.mean(values))
                error_val = float(np.std(values, ddof=0)) if values.size > 1 else 0.0
        elif global_df is not None and global_mode == "cv_across_seeds":
            df2 = global_df.copy()
            if "metric" in df2.columns:
                df2["metric"] = df2["metric"].astype(str).str.strip().str.lower()
                row = df2[(df2["model"] == model) & (df2["metric"] == metric_name)]
                if not row.empty and "mean" in row.columns and "std" in row.columns:
                    mean_val = float(row.iloc[0]["mean"])
                    error_val = float(row.iloc[0]["std"])
                else:
                    mean_val = float(np.mean(values))
                    error_val = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            else:
                mean_val = float(np.mean(values))
                error_val = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        else:
            mean_val = float(np.mean(values))
            if error_type == "std":
                ddof = 1 if stage == "cv" else 0
                error_val = float(np.std(values, ddof=ddof)) if values.size > 1 else 0.0
            elif error_type == "ci95":
                error_val = ci95_from_scalars(values.tolist())
            else:
                error_val = 0.0
        
        agg_data.append({
            "model": model,
            "mean": mean_val,
            "error": error_val,
        })
    
    return pd.DataFrame(agg_data)


# %%
# Plot A: Back-to-Back Bar Chart
def plot_back_to_back_bars(
    base_dir: Path,
    metric_df: pd.DataFrame,
    stage: str,
    output_dir: Path,
    dpi: int,
    font: str,
    global_color_map: Dict[str, tuple],
    error_type: str = "std",
) -> None:
    """Plot A: Clean, non-overlapping publication-ready back-to-back bar chart.

    Features descending MCC sorting, bottom-aligned X-axis labels, 
    and explicit Cambria font rendering to avoid cross-plot interference.
    """
    if metric_df.empty:
        return
    
    import matplotlib.pyplot as plt
    configure_plotting(font)
    
    
    # 2. Data calculation and strict descending sort by MCC mean
    mcc_stats = []
    models_list = metric_df["model"].unique()
    
    # Prefer explicit summary CSVs when available so numbers match tables exactly.
    global_df = None
    global_mode = None
    if error_type == "std":
        if stage == "cv":
            cv_path = base_dir / "results" / "all_seed_cv_summary_across_seeds.csv"
            if cv_path.exists():
                global_df = pd.read_csv(cv_path)
                global_mode = "cv_across_seeds"
        else:
            ext_path = base_dir / "results" / "all_seed_external_summary.csv"
            if ext_path.exists():
                global_df = pd.read_csv(ext_path)
                global_mode = "external_global"

    for model in models_list:
        m_data = metric_df[(metric_df["model"] == model) & (metric_df["metric"] == "mcc")]
        if not m_data.empty:
            values = m_data["value"].to_numpy()
            
            if global_df is not None and global_mode == "external_global":
                row = global_df[global_df["model"] == model]
                if not row.empty and "mcc_mean" in row.columns and "mcc_std" in row.columns:
                    mean_val = float(row.iloc[0]["mcc_mean"])
                    error_val = float(row.iloc[0]["mcc_std"])
                else:
                    mean_val = float(np.mean(values))
                    error_val = float(np.std(values, ddof=0)) if values.size > 1 else 0.0
            elif global_df is not None and global_mode == "cv_across_seeds":
                df2 = global_df.copy()
                if "metric" in df2.columns:
                    df2["metric"] = df2["metric"].astype(str).str.strip().str.lower()
                    row = df2[(df2["model"] == model) & (df2["metric"] == "mcc")]
                    if not row.empty:
                        mean_val = float(row.iloc[0]["mean"])
                        error_val = float(row.iloc[0]["std"])
                    else:
                        mean_val = float(np.mean(values))
                        error_val = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
                else:
                    mean_val = float(np.mean(values))
                    error_val = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            else:
                mean_val = float(np.mean(values))
                if error_type == "std":
                    ddof = 1 if stage == "cv" else 0
                    error_val = float(np.std(values, ddof=ddof)) if values.size > 1 else 0.0
                elif error_type == "ci95":
                    error_val = ci95_from_scalars(values.tolist())
                else:
                    error_val = 0.0
            mcc_stats.append({"model": model, "mcc_mean": mean_val, "mcc_error": error_val})
    
    # 降序排列：MCC 从大到小
    mcc_df = pd.DataFrame(mcc_stats).sort_values("mcc_mean", ascending=False).reset_index(drop=True)
    
    # 为了让 Matplotlib 的 barh 从上往下画时呈现降序，将 DataFrame 逆序传入
    mcc_df_plot = mcc_df.iloc[::-1].reset_index(drop=True)
    
    models = mcc_df_plot["model"].tolist()
    y_pos = np.arange(len(models))
    colors = [global_color_map.get(m, (0.7, 0.7, 0.7)) for m in models]

    # 3. 构建画布：为中央走廊和下方的 X 轴标签留出刚好合适的间距
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11.5, 6.0), sharey=True, dpi=dpi)
    
    # 明确物理归位两个子图的坐标朝向，防止被图 D 更改的全局设置带偏
    ax_left.yaxis.tick_left()
    ax_right.yaxis.tick_left()
    ax_left.xaxis.tick_bottom()
    ax_right.xaxis.tick_bottom()
    
    plt.subplots_adjust(wspace=0.26, top=0.88, bottom=0.15) 

    bar_height = 0.85 

    # 4. 绘制左侧：MCC RANKING (条形向左延伸)
    stage_upper = "CROSS-VALIDATION" if stage.lower() == "cv" else "EXTERNAL TEST"
    bars_left = ax_left.barh(y_pos, mcc_df_plot["mcc_mean"], bar_height, color=colors, alpha=0.9, edgecolor='black', linewidth=0.8)
    
    # 渲染子图标题
    ax_left.set_title(f"{stage_upper}: MCC Ranking", 
                      fontsize=11, fontweight='bold', pad=15)
    ax_left.invert_xaxis()  # 反转 X 轴（1.0 在左，0 在右中央）
    ax_left.set_xlim(1.0, 0) 
    ax_left.grid(axis='x', linestyle='--', alpha=0.3, zorder=0)
    
    # 将 MCC 数值精准嵌入左侧 Bar 内部
    for i, val in enumerate(mcc_df_plot["mcc_mean"]):
        ax_left.text(val - 0.02, i, f"{val:.{LEGEND_DECIMALS}f}", va='center', ha='left',
                     fontsize=10, fontweight='bold', color="#111111", zorder=5)

    # 5. 绘制右侧：STABILITY RANKING (条形向右延伸)
    error_label = "STD" if error_type == "std" else "CI95"
    bars_right = ax_right.barh(y_pos, mcc_df_plot["mcc_error"], bar_height, color=colors, alpha=0.9, edgecolor='black', linewidth=0.8)
    
    # 渲染子图标题
    ax_right.set_title(f"{stage_upper}: Stability Ranking", 
                       fontsize=11, fontweight='bold', pad=15)
    
    # 安全缩减空白，卡在最大值的1.12倍
    max_err = mcc_df_plot["mcc_error"].max()
    ax_right.set_xlim(0, max_err * 1.12 + 0.005)
    ax_right.grid(axis='x', linestyle='--', alpha=0.3, zorder=0)
    
    # 将 Stability 数值嵌入右侧 Bar 内部
    for i, val in enumerate(mcc_df_plot["mcc_error"]):
        ax_right.text(val - 0.003, i, f"{val:.{LEGEND_DECIMALS}f}", va='center', ha='right',
                     fontsize=10, fontweight='bold', color="#111111", zorder=5)

    # 6. 配置核心 Y 轴刻度位置与标签显示
    ax_left.set_yticks(y_pos)
    ax_left.set_yticklabels([]) # 移除轴线默认自带的边缘标签文本
    
    # 显式修正中央走廊两侧的刻度线挂载
    ax_left.yaxis.tick_right()
    ax_right.yaxis.tick_left()

    # 在中央走廊的正中心绘制模型名称
    for i, model_name in enumerate(models):
        ax_left.text(-0.13, i, model_name, ha='center', va='center', 
                     fontsize=11, fontweight='bold', color='#222222')

    # 中央走廊顶部追加列名标识 "Model Name"
    ax_left.text(-0.13, len(models) - 0.3, "Model Name", ha='center', va='bottom', 
                 fontsize=11, fontweight='bold', color='black')

    # 7. 确保 X 轴坐标标签严格放置在最下方（Bottom）
    ax_left.xaxis.set_ticks_position('bottom')
    ax_right.xaxis.set_ticks_position('bottom')
    ax_left.xaxis.set_label_position('bottom')
    ax_right.xaxis.set_label_position('bottom')
    
    ax_left.set_xlabel("Mathews Correlation Coefficient (MCC)", fontsize=10, fontweight='bold', labelpad=8)
    ax_right.set_xlabel(f"Stability Standard Deviation ({error_label})", fontsize=10, fontweight='bold', labelpad=8)

    # 8. 坐标轴线条可见性精细打磨（为左右两图加上全包围黑色边框）
    for ax in [ax_left, ax_right]:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)
            spine.set_color('black')

    # 标出 Y 轴刻度短线：向中央走廊的外侧指示对应模型
    ax_left.tick_params(axis='y', which='both', direction='out', length=4, width=1.2, color='black')
    ax_right.tick_params(axis='y', which='both', direction='out', length=4, width=1.2, color='black')
    
    # 确保 X 轴刻度短线也足够清晰
    ax_left.tick_params(axis='x', which='both', direction='out', length=4, width=1.2, color='black')
    ax_right.tick_params(axis='x', which='both', direction='out', length=4, width=1.2, color='black')

    # 9. 保存矢量图结构
    output_path = output_dir / f"{stage}_plot_A_back_to_back.svg"
    fig.savefig(output_path, format="svg", bbox_inches='tight')
    print(f"[OK] Saved clean presentation-ready plot A with strict Cambria typography: {output_path}")
    plt.close(fig)

# %%
# Plot B: Dot Plot with Error Bars

def plot_dot_with_error_bars(
    base_dir: Path,
    metric_df: pd.DataFrame,
    stage: str,
    output_dir: Path,
    dpi: int,
    font: str,
    global_color_map: Dict[str, tuple],
    error_type: str = "std",
) -> None:
    if metric_df.empty: return
    configure_plotting(font)
    
    mcc_df = aggregate_metrics_by_model(base_dir, stage, metric_df, "mcc", error_type)
    if mcc_df.empty: return
    
    mcc_df = mcc_df.sort_values("mean", ascending=False).reset_index(drop=True)
    models = mcc_df["model"].tolist()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(models))
    
    for i, row in mcc_df.iterrows():
        model = row["model"]
        color = global_color_map.get(model, (0.5, 0.5, 0.5))
        ax.errorbar(
            row["mean"], i, xerr=row["error"], fmt="o", markersize=10,
            color=color, ecolor="black", elinewidth=1.5, capsize=4, capthick=1.5,
            markeredgecolor="black", markeredgewidth=1.0, zorder=3
        )
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontweight='bold')
    ax.set_xlabel("MCC (Mean ± Error)")
    ax.set_ylabel("Model")
    
    stage_label = "Cross-Validation" if stage.lower() == "cv" else "External Test"
    error_label = "Std" if error_type == "std" else "CI95"
    ax.set_title(f"(B) {stage_label}: MCC with {error_label}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(0, 1.0)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    
    fig.tight_layout()
    output_path = output_dir / f"{stage}_plot_B_dot_error_bars.svg"
    fig.savefig(output_path, format="svg")
    print(f"[OK] Saved plot B (dot error bars): {output_path}")
    plt.close(fig)


# %%
# Plot C: 2D Performance Space

def plot_2d_performance_space(
    base_dir: Path,
    metric_df: pd.DataFrame,
    stage: str,
    output_dir: Path,
    dpi: int,
    font: str,
    global_color_map: Dict[str, tuple],
    error_type: str = "std",
) -> None:
    if metric_df.empty: return
    configure_plotting(font)
    
    mcc_df = aggregate_metrics_by_model(base_dir, stage, metric_df, "mcc", error_type)
    if mcc_df.empty: return
    
    models = mcc_df["model"].tolist()
    mcc_means = mcc_df["mean"].values
    
    use_global_std = (error_type == "std")
    global_df = None
    if use_global_std:
        global_path = base_dir / "results" / (
            "all_seed_cv_summary.csv" if stage == "cv" else "all_seed_external_summary.csv"
        )
        if global_path.exists():
            global_df = pd.read_csv(global_path)

    stab_means = []
    for model in models:
        model_data = metric_df[metric_df["model"] == model]
        stds = []
        for metric in ["mcc", "f1", "accuracy", "precision", "recall"]:
            if use_global_std and global_df is not None:
                if stage == "cv":
                    row = global_df[(global_df["model"] == model) & (global_df["metric"] == metric)]
                    if not row.empty:
                        stds.append(float(row.iloc[0]["std"]))
                    else:
                        sub = model_data[model_data["metric"] == metric]["value"].to_numpy()
                        if len(sub) > 1: stds.append(np.std(sub, ddof=1))
                else:
                    row = global_df[global_df["model"] == model]
                    if not row.empty and f"{metric}_std" in row.columns:
                        stds.append(float(row.iloc[0][f"{metric}_std"]))
                    else:
                        sub = model_data[model_data["metric"] == metric]["value"].to_numpy()
                        if len(sub) > 1: stds.append(np.std(sub, ddof=1))
            else:
                sub = model_data[model_data["metric"] == metric]["value"].to_numpy()
                if len(sub) > 1: stds.append(np.std(sub, ddof=1))
        stab_means.append(np.mean(stds) if stds else 0.0)
    stab_means = np.array(stab_means)
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
    
    for i, model in enumerate(models):
        color = global_color_map.get(model, (0.5, 0.5, 0.5))
        ax.scatter(mcc_means[i], stab_means[i], s=350, color=color, edgecolor="black", linewidth=1.2, zorder=3, alpha=0.85)
        
        # Fix: High-saturation blue for SVC uses white text; other light Morandi colors use black text
        text_color = "white" if model == "SVC" else "black"
        ax.text(mcc_means[i], stab_means[i], model, ha="center", va="center", fontsize=9, fontweight="bold", color=text_color, zorder=4)
    
    mcc_threshold = np.percentile(mcc_means, 50) if len(mcc_means) > 1 else 0.5
    std_threshold = np.percentile(stab_means, 50) if len(stab_means) > 1 else 0.1
    
    ax.axvspan(mcc_threshold, 1.0, alpha=0.06, color="green")
    ax.axhspan(0, std_threshold, alpha=0.06, color="blue")
    
    ax.text(0.95, 0.05, "* Sweet Spot", transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "#FFFFCC", "edgecolor": "grey", "alpha": 0.8})
    
    ax.set_xlabel("MCC (Mean)", fontweight='bold')
    ax.set_ylabel("Stability (Mean Std Across Metrics)", fontweight='bold')
    stage_label = "Cross-Validation" if stage.lower() == "cv" else "External Test"
    ax.set_title(f"(C) {stage_label}: 2D Performance Space", fontsize=13, fontweight="bold", pad=12)
    
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_xlim(min(mcc_means.min() * 0.9, 0.4), 1.0)
    ax.set_ylim(0, max(stab_means) * 1.3 if max(stab_means) > 0 else 0.2)
    
    fig.tight_layout()
    output_path = output_dir / f"{stage}_plot_C_2d_performance.svg"
    fig.savefig(output_path, format="svg")
    print(f"[OK] Saved plot C (2D performance): {output_path}")
    plt.close(fig)


# %%
# Plot D: Model Characteristics Heatmap

def plot_model_heatmap(
    base_dir: Path,
    metric_df: pd.DataFrame,
    stage: str,
    output_dir: Path,
    dpi: int,
    font: str,
    global_color_map: Dict[str, tuple],
) -> None:
    """Plot D: Presentation-ready Model Heatmap using a custom Morandi sequential

    colormap with an explicit black outer border and unified black text labels.
    """
    if metric_df.empty: 
        return
        
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import matplotlib.colors as mcolors
    
    configure_plotting(font)
    
    metrics_to_plot = ["mcc", "f1", "accuracy", "precision", "recall"]
    
    # 2. Strict descending sort by MCC mean (ensure best performing models are at the top)
    mcc_order = (
        metric_df[metric_df["metric"] == "mcc"]
        .groupby("model")["value"]
        .mean()
        .sort_values(ascending=False)
        .index
    )
    models = list(mcc_order)
    
    # 3. Extract the true raw data matrix
    raw_values = []
    for model in models:
        row = []
        for metric in metrics_to_plot:
            val = metric_df[(metric_df["model"] == model) & (metric_df["metric"] == metric)]["value"].mean()
            row.append(val if not np.isnan(val) else 0.0)
        raw_values.append(row)
    raw_values = np.array(raw_values)

    # 4. Column-wise Min-Max normalization to clearly separate intra-metric distinctions
    norm_matrix = np.zeros_like(raw_values)
    for j in range(len(metrics_to_plot)):
        col = raw_values[:, j]
        col_min, col_max = col.min(), col.max()
        if col_max > col_min:
            norm_matrix[:, j] = (col - col_min) / (col_max - col_min)
        else:
            norm_matrix[:, j] = 1.0

    # 5. Build exclusive Morandi continuous colormap (Pink -> Dark Gold -> Core Highlight Blue)
    morandi_colors = ["#B08B86", "#C3B083", "#1F77B4"]
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("MorandiPerformance", morandi_colors, N=256)

    # 6. Create clean canvas
    fig, ax = plt.subplots(figsize=(8.8, 5.5), dpi=dpi)
    ax.yaxis.tick_left()
    ax.xaxis.tick_bottom()
    
    # Render heatmap base
    im = ax.imshow(norm_matrix, cmap=custom_cmap, aspect="auto", interpolation='nearest', vmin=0.0, vmax=1.0)
    
    # 7. Manually draw inner grid lines for cells
    for i in range(len(models)):
        for j in range(len(metrics_to_plot)):
            val = raw_values[i, j]
            
            # Retain fine white border lines between inner cells to form a chocolate-bar grid structure
            rect = Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='white', linewidth=1.5)
            ax.add_patch(rect)
            
            ax.text(j, i, f"{val:.{LEGEND_DECIMALS}f}", ha="center", va="center",
                    color="#000000", fontsize=11, fontweight="bold")

    # 8. [Core modification]: Force a solid black outer border around the outermost edge of the heatmap matrix
    outer_rect = Rectangle((-0.5, -0.5), len(metrics_to_plot), len(models), 
                           fill=False, edgecolor='#000000', linewidth=1.5, zorder=10)
    ax.add_patch(outer_rect)

    # 9. Precisely configure coordinate grid and ranges (strongest model at the top)
    ax.set_xlim(-0.5, len(metrics_to_plot) - 0.5)
    ax.set_ylim(len(models) - 0.5, -0.5)
    
    ax.set_xticks(np.arange(len(metrics_to_plot)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels([m.upper() for m in metrics_to_plot], fontsize=11, fontweight='bold')
    ax.set_yticklabels(models, fontsize=11, fontweight='bold')
    
    # Turn off Matplotlib's default outer spines, as we perfectly drew a pure black border using Rectangle in the previous step
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(top=False, bottom=False, left=False, right=False)

    # 10. Normalize single-layer main title
    stage_upper = "CROSS-VALIDATION" if stage.lower() == "cv" else "EXTERNAL TEST"
    ax.set_title(f"{stage_upper}: Model Performance Evaluation Heatmap", 
                 pad=18, fontsize=12, fontweight="bold")
    
    # 11. Beautify right-side continuous Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    '''
    cbar.set_label("Relative Performance Score", 
                   fontsize=10, fontweight='bold', labelpad=10)
    '''
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0)
    
    # Label both ends of the Colorbar
    cbar.set_ticks([0.02, 0.98])
    cbar.set_ticklabels(["Lowest\nPerformance", "Highest\nPerformance"],fontweight='bold')
    
    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_fontsize(9.5)

    plt.tight_layout()

    # 12. Save high-precision vector graphics
    output_path = output_dir / f"{stage}_plot_D_heatmap.svg"
    fig.savefig(output_path, format="svg", bbox_inches='tight')
    print(f"[OK] Saved plot D with black outer border and black text: {output_path}")
    plt.close(fig)


# %%
# Plot E: Feature Importance Radar Chart

def plot_radar_chart(
    base_dir: Path,
    metric_df: pd.DataFrame,
    stage: str,
    output_dir: Path,
    dpi: int,
    font: str,
    global_color_map: Dict[str, tuple],
) -> None:
    if metric_df.empty: return
    configure_plotting(font)
    
    metrics_to_plot = ["mcc", "f1", "accuracy", "precision", "recall"]
    models = sorted(metric_df["model"].unique())
    
    model_metrics = {}
    for model in models:
        values = []
        for metric in metrics_to_plot:
            sub_df = metric_df[(metric_df["model"] == model) & (metric_df["metric"] == metric)]
            value = sub_df["value"].mean() if not sub_df.empty else 0.5
            values.append(np.clip(value, 0, 1))
        model_metrics[model] = values
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"), dpi=dpi)
    
    angles = np.linspace(0, 2 * np.pi, len(metrics_to_plot), endpoint=False).tolist()
    angles += angles[:1]
    
    for model in models:
        values = model_metrics[model] + model_metrics[model][:1]
        color = global_color_map.get(model, (0.5, 0.5, 0.5))
        ax.plot(angles, values, "o-", linewidth=2.0, color=color, label=model, markersize=6, markeredgecolor="black", markeredgewidth=0.4)
        ax.fill(angles, values, color=color, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.upper() for m in metrics_to_plot], fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    
    stage_label = "Cross-Validation" if stage.lower() == "cv" else "External Test"
    ax.set_title(f"(E) {stage_label}: Model Performance Radar Chart", fontsize=13, fontweight="bold", pad=20)
    
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.05), frameon=True, edgecolor="black", facecolor="white")
    
    fig.tight_layout()
    output_path = output_dir / f"{stage}_plot_E_radar.svg"
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    print(f"[OK] Saved plot E (radar): {output_path}")
    plt.close(fig)


# %%
def save_as_tiff(svg_path: Path, tiff_path: Path, dpi: int = 600) -> None:
    try:
        import subprocess
        subprocess.run(["convert", "-density", str(dpi), str(svg_path), str(tiff_path)], check=True, capture_output=True)
        print(f"[OK] Converted to TIFF: {tiff_path}")
    except Exception:
        pass


# %%
def main() -> None:
    args = parse_args()
    
    try:
        base_dir = resolve_base_dir(args.base_dir)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return
    
    output_dir = args.output_dir or (base_dir / "figures" / "enhanced")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stages = []
    if args.include_external or "external" in args.stages: stages.append("external")
    if args.include_cv or "cv" in args.stages: stages.append("cv")
    if not stages: stages = args.stages
    
    print(f"[INFO] Base run directory: {base_dir}")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Stages: {stages}")
    print(f"[INFO] Error type: {args.error_type}")
    print(f"[INFO] Plot types: {args.plot_types}")
    
    global_color_map: Dict[str, tuple] = {}
    
    for stage in stages:
        print(f"\n[INFO] Processing stage: {stage}")
        metric_df = prepare_metric_dataframe(base_dir, stage, DEFAULT_METRICS)
        if metric_df.empty:
            print(f"[WARN] No data found for stage '{stage}'")
            continue
        
        models = sorted(metric_df["model"].unique())
        get_or_assign_model_colors(models, args.palette, global_color_map)
        
        if "A" in args.plot_types:
            plot_back_to_back_bars(base_dir, metric_df, stage, output_dir, args.dpi, args.font, global_color_map, args.error_type)
        if "B" in args.plot_types:
            plot_dot_with_error_bars(base_dir, metric_df, stage, output_dir, args.dpi, args.font, global_color_map, args.error_type)
        if "C" in args.plot_types:
            plot_2d_performance_space(base_dir, metric_df, stage, output_dir, args.dpi, args.font, global_color_map, args.error_type)
        if "D" in args.plot_types:
            plot_model_heatmap(base_dir, metric_df, stage, output_dir, args.dpi, args.font, global_color_map)
        if "E" in args.plot_types:
            plot_radar_chart(base_dir, metric_df, stage, output_dir, args.dpi, args.font, global_color_map)
        
        if args.export_tiff:
            svg_files = list(output_dir.glob(f"{stage}_plot_*.svg"))
            for svg_path in svg_files:
                save_as_tiff(svg_path, svg_path.with_suffix(".tiff"), args.dpi)
    
    print(f"\n[INFO] All plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
