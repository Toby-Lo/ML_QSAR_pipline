#!/usr/bin/env python3
"""
Step36: Weight sensitivity test for the final screening score.

Usage:
  python scripts/step36_weight_sensitivity_test.py \
    --input-parquet models_out/qsar_ml_20260412_162829/virtual_screening/ad_screening_results_20260419_140044/admet_scoring_20260419_173114/admet_scored.parquet

  python scripts/step36_weight_sensitivity_test.py \
    --input-parquet models_out/qsar_ml_20260412_162829/virtual_screening/ad_screening_results_20260419_140044/admet_scoring_20260419_173114/admet_scored.parquet \
    --output-dir models_out/qsar_ml_20260412_162829/virtual_screening/ad_screening_results_20260419_140044/admet_scoring_20260419_173114/ \
    --top-ns 50,100,500

Default weight schemes tested:
  - 0.5 / 0.2 / 0.3
  - 0.6 / 0.2 / 0.2
  - 0.4 / 0.3 / 0.3
  - 0.5 / 0.3 / 0.2

Outputs:
  - weight_sensitivity_rankings.csv
  - weight_sensitivity_pairwise_overlap_top*.csv
  - weight_sensitivity_overlap_matrix_top*.csv
  - weight_sensitivity_top_membership_top*.csv
  - weight_sensitivity_overlap_heatmap_top*.png / .svg
  - weight_sensitivity_overlap_heatmap_combined.png / .svg
  - weight_sensitivity_summary_top*.json
  - weight_sensitivity_summary.json

check values:
import pandas as pd
df = pd.read_csv("models_out/qsar_ml_20260412_162829/virtual_screening/ad_screening_results_20260419_140044/admet_scoring_20260419_173114/weight_sensitivity_top_membership_top100.csv")
union_size = len(df)
intersection_size = len(df[df['n_schemes'] == 4])
consensus_ratio = (intersection_size / 100) * 100

print(f"Intersection Size (Common): {intersection_size}")
print(f"Union Size: {union_size}")
print(f"Consensus Ratio: {consensus_ratio}%")

common_compounds = len(df[df['n_schemes'] == 4]) 
print(f"Common Compounds (Intersection): {common_compounds}") 

union_all = len(df)
print(f"Union of All Schemes: {union_all}")

consensus_ratio = (common_compounds / 100) * 100
print(f"Consensus Ratio: {consensus_ratio:.1f}%")

df_pairwise = pd.read_csv("models_out/qsar_ml_20260412_162829/virtual_screening/ad_screening_results_20260419_140044/admet_scoring_20260419_173114/weight_sensitivity_pairwise_overlap_top100.csv")
min_overlap = df_pairwise['overlap_ratio'].min() * 100
print(f"Minimum Pairwise Overlap: {min_overlap:.1f}%")
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


WEIGHT_SCHEMES: Tuple[Tuple[str, Dict[str, float]], ...] = (
    ("qsar_0p5_ad_0p2_admet_0p3", {"qsar_prob": 0.5, "ad_score": 0.2, "admet_score": 0.3}),
    ("qsar_0p6_ad_0p2_admet_0p2", {"qsar_prob": 0.6, "ad_score": 0.2, "admet_score": 0.2}),
    ("qsar_0p4_ad_0p3_admet_0p3", {"qsar_prob": 0.4, "ad_score": 0.3, "admet_score": 0.3}),
    ("qsar_0p5_ad_0p3_admet_0p2", {"qsar_prob": 0.5, "ad_score": 0.3, "admet_score": 0.2}),
)

DEFAULT_REQUIRED_COLUMNS: Tuple[str, ...] = ("qsar_prob", "ad_score", "admet_score")
DEFAULT_IDENTIFIER_COLUMNS: Tuple[str, ...] = (
    "smiles",
    "raw_smiles",
    "canonical_smiles",
    "canon_smiles",
    "compound_id",
    "molecule_id",
    "ligand_id",
    "name",
    "title",
)


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("step36_weight_sensitivity")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sensitivity analysis for Step35 final_score weighting.")
    parser.add_argument(
        "--input-parquet",
        type=Path,
        default=None,
        help="Step35 output parquet containing qsar_prob, ad_score, and admet_score.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for CSV/figure outputs. Default: <input_parent>/weight_sensitivity_<timestamp>.",
    )
    parser.add_argument(
        "--top-ns",
        type=str,
        default="100",
        help="Comma-separated Top-N cutoffs to compare across weighting schemes (default: 100). Example: 50,100,500",
    )
    parser.add_argument(
        "--id-col",
        type=str,
        default=None,
        help="Identifier column for compounds. Default: first match among common ID/SMILES columns.",
    )
    parser.add_argument(
        "--no-make-plots",
        dest="make_plots",
        action="store_false",
        help="Skip the overlap heatmap figure.",
    )
    parser.set_defaults(make_plots=True)
    return parser.parse_args(argv)


def resolve_input_parquet(path: Optional[Path]) -> Path:
    if path is not None:
        resolved = path.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Input parquet not found: {resolved}")
        return resolved

    candidates = sorted(Path("models_out").glob("**/admet_scored.parquet"), key=lambda p: p.as_posix())
    if not candidates:
        raise FileNotFoundError(
            "No admet_scored.parquet found under models_out. Provide --input-parquet explicitly."
        )
    return candidates[-1].resolve()


def resolve_identifier_col(df: pd.DataFrame, requested: Optional[str]) -> str:
    if requested is not None:
        if requested not in df.columns:
            raise KeyError(f"Requested id column not found: {requested}")
        return requested

    for col in DEFAULT_IDENTIFIER_COLUMNS:
        if col in df.columns:
            return col
    return "__row_id__"


def parse_top_ns(value: str) -> List[int]:
    top_ns: List[int] = []
    for part in str(value).split(","):
        cleaned = part.strip()
        if not cleaned:
            continue
        top_n = int(cleaned)
        if top_n <= 0:
            raise ValueError("Top-N values must be positive integers.")
        top_ns.append(top_n)
    if not top_ns:
        raise ValueError("At least one Top-N value is required.")
    return sorted(set(top_ns))


def minmax_01(x: pd.Series) -> pd.Series:
    values = pd.to_numeric(x, errors="coerce").astype("float64")
    finite = values[np.isfinite(values)]
    if finite.empty:
        return pd.Series(np.nan, index=values.index, dtype="float64")
    lo = float(finite.min())
    hi = float(finite.max())
    if np.isclose(lo, hi):
        return pd.Series(0.5, index=values.index, dtype="float64")
    return ((values - lo) / (hi - lo)).clip(0.0, 1.0)


def sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def get_scheme_label_map() -> Dict[str, str]:
    return {
        "qsar_0p5_ad_0p2_admet_0p3": "0.5 / 0.2 / 0.3",
        "qsar_0p6_ad_0p2_admet_0p2": "0.6 / 0.2 / 0.2",
        "qsar_0p4_ad_0p3_admet_0p3": "0.4 / 0.3 / 0.3",
        "qsar_0p5_ad_0p3_admet_0p2": "0.5 / 0.3 / 0.2",
    }


def prepare_display_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    scheme_labels = get_scheme_label_map()
    display_matrix = matrix.copy()
    display_matrix.index = [scheme_labels.get(label, label) for label in matrix.index]
    display_matrix.columns = [scheme_labels.get(label, label) for label in matrix.columns]
    return display_matrix


def ordered_unique(items: Sequence[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def build_weighted_scores(df: pd.DataFrame, scheme_name: str, weights: Dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    raw = pd.Series(0.0, index=out.index, dtype="float64")
    for col, weight in weights.items():
        raw = raw + pd.to_numeric(out[col], errors="coerce").astype("float64") * float(weight)
    out["weight_scheme"] = scheme_name
    out["final_score_raw"] = raw.astype("float64")
    out["final_score"] = minmax_01(out["final_score_raw"]).astype("float64")
    out = out.sort_values(["final_score_raw", "__sort_id__"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1, dtype=np.int64)
    return out


def overlap_summary(top_sets: Dict[str, set], top_n: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for left, right in combinations(top_sets.keys(), 2):
        left_set = top_sets[left]
        right_set = top_sets[right]
        overlap = len(left_set & right_set)
        union = len(left_set | right_set)
        denom = min(len(left_set), len(right_set)) or int(top_n)
        rows.append(
            {
                "scheme_left": left,
                "scheme_right": right,
                "top_n": int(top_n),
                "overlap_count": int(overlap),
                "overlap_ratio": float(overlap / denom) if denom else np.nan,
                "jaccard": float(overlap / union) if union else np.nan,
            }
        )
    return pd.DataFrame(rows)


def overlap_matrix(top_sets: Dict[str, set], top_n: int) -> pd.DataFrame:
    schemes = list(top_sets.keys())
    matrix = pd.DataFrame(index=schemes, columns=schemes, dtype="float64")
    for left in schemes:
        for right in schemes:
            overlap = len(top_sets[left] & top_sets[right])
            denom = min(len(top_sets[left]), len(top_sets[right])) or int(top_n)
            matrix.loc[left, right] = float(overlap / denom) if denom else np.nan
    return matrix


def top_membership_table(
    ranked_tables: Dict[str, pd.DataFrame],
    top_sets: Dict[str, set],
    id_col: str,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    all_top_ids = sorted(set().union(*top_sets.values()))
    for compound_id in all_top_ids:
        row: Dict[str, object] = {id_col: compound_id}
        in_schemes: List[str] = []
        ranks: List[int] = []
        scores: List[float] = []
        for scheme_name, table in ranked_tables.items():
            hit = table.loc[table[id_col] == compound_id]
            if hit.empty:
                continue
            in_top = compound_id in top_sets[scheme_name]
            if in_top:
                in_schemes.append(scheme_name)
            rank_value = int(hit["rank"].iloc[0])
            score_value = float(hit["final_score"].iloc[0])
            ranks.append(rank_value)
            scores.append(score_value)
            row[f"rank_{sanitize_name(scheme_name)}"] = rank_value
            row[f"score_{sanitize_name(scheme_name)}"] = score_value
            row[f"in_top_{sanitize_name(scheme_name)}"] = bool(in_top)
        row["n_schemes"] = len(in_schemes)
        row["schemes"] = ";".join(in_schemes)
        row["best_rank"] = int(min(ranks)) if ranks else np.nan
        row["mean_final_score"] = float(np.mean(scores)) if scores else np.nan
        rows.append(row)
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["n_schemes", "best_rank", id_col], ascending=[False, True, True], kind="mergesort")
    return result


def plot_overlap_heatmap(matrix: pd.DataFrame, output_dir: Path, top_n: int) -> Tuple[Path, Path]:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Cambria", "Times New Roman", "DejaVu Serif"],
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
        }
    )

    display_matrix = prepare_display_matrix(matrix)

    morandi_cmap = LinearSegmentedColormap.from_list(
        "morandi_overlap",
        ["#f5efe8", "#ddd8cc", "#c8c6be", "#b0bac2", "#8fa3b2"],
        N=256,
    )
    norm = Normalize(vmin=0.7, vmax=1.0)

    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    data = display_matrix.to_numpy(dtype="float64")
    im = ax.imshow(data, norm=norm, cmap=morandi_cmap)
    ax.set_xticks(np.arange(display_matrix.shape[1]))
    ax.set_yticks(np.arange(display_matrix.shape[0]))
    ax.set_xticklabels(display_matrix.columns.tolist(), rotation=0, ha="center", fontweight="bold")
    ax.set_yticklabels(display_matrix.index.tolist(), fontweight="bold")
    ax.set_title(f"Top-{top_n} overlap across weight schemes", fontweight="bold", pad=14)
    ax.set_xlabel("Weighting scheme", fontweight="bold", labelpad=10)
    ax.set_ylabel("Weighting scheme", fontweight="bold", labelpad=10)

    ax.set_xticks(np.arange(-0.5, display_matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, display_matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="x", pad=12)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = data[i, j]
            rgba = morandi_cmap(norm(value))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            text_color = "white" if luminance < 0.58 else "black"
            ax.text(
                j,
                i,
                f"{value:.0%}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
                fontweight="bold",
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Overlap ratio", rotation=90, fontweight="bold")
    cbar.ax.tick_params(labelsize=10)
    fig.tight_layout()

    png_path = output_dir / "weight_sensitivity_overlap_heatmap.png"
    svg_path = output_dir / "weight_sensitivity_overlap_heatmap.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path


def plot_combined_overlap_heatmaps(
    matrices_by_top_n: Dict[int, pd.DataFrame],
    output_dir: Path,
) -> Tuple[Path, Path]:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Cambria", "Times New Roman", "DejaVu Serif"],
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
        }
    )

    morandi_cmap = LinearSegmentedColormap.from_list(
        "morandi_overlap",
        ["#f5efe8", "#ddd8cc", "#c8c6be", "#b0bac2", "#8fa3b2"],
        N=256,
    )
    norm = Normalize(vmin=0.7, vmax=1.0)

    top_ns = list(matrices_by_top_n.keys())
    n_panels = len(top_ns)
    fig = plt.figure(figsize=(5.9 * n_panels + 0.8, 6.0))
    grid = fig.add_gridspec(1, n_panels + 1, width_ratios=[1.0] * n_panels + [0.045], wspace=0.10)
    axes = [fig.add_subplot(grid[0, idx]) for idx in range(n_panels)]
    cax = fig.add_subplot(grid[0, -1])

    im = None
    for ax, top_n in zip(axes, top_ns):
        display_matrix = prepare_display_matrix(matrices_by_top_n[top_n])
        data = display_matrix.to_numpy(dtype="float64")
        im = ax.imshow(data, norm=norm, cmap=morandi_cmap)
        ax.set_xticks(np.arange(display_matrix.shape[1]))
        ax.set_yticks(np.arange(display_matrix.shape[0]))
        ax.set_xticklabels(display_matrix.columns.tolist(), rotation=0, ha="center", fontweight="bold", fontsize=10)
        ax.set_yticklabels(display_matrix.index.tolist(), fontweight="bold", fontsize=10)
        ax.set_title(f"Top-{top_n}", fontweight="bold", pad=12, fontsize=15)
        ax.set_xlabel("Weighting scheme", fontweight="bold", labelpad=10)
        if ax is axes[0]:
            ax.set_ylabel("Weighting scheme", fontweight="bold", labelpad=10)
        else:
            ax.set_ylabel("")

        ax.set_xticks(np.arange(-0.5, display_matrix.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, display_matrix.shape[0], 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.tick_params(axis="x", pad=12)

        for i in range(display_matrix.shape[0]):
            for j in range(display_matrix.shape[1]):
                value = data[i, j]
                rgba = morandi_cmap(norm(value))
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                text_color = "white" if luminance < 0.58 else "black"
                ax.text(
                    j,
                    i,
                    f"{value:.0%}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=9.5,
                    fontweight="bold",
                )

    if im is None:
        raise ValueError("No matrices provided for combined heatmap plotting.")

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Overlap ratio", rotation=90, fontweight="bold")
    cbar.ax.tick_params(labelsize=10)
    fig.suptitle("Top-N overlap across weight schemes", fontweight="bold", fontsize=16, y=1.02)
    fig.subplots_adjust(left=0.03, right=0.985, top=0.90, bottom=0.10, wspace=0.08)

    png_path = output_dir / "weight_sensitivity_overlap_heatmap_combined.png"
    svg_path = output_dir / "weight_sensitivity_overlap_heatmap_combined.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logger = setup_logger()

    input_path = resolve_input_parquet(args.input_parquet)
    df = pd.read_parquet(input_path)

    missing = [col for col in DEFAULT_REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required score columns in {input_path}: {missing}")

    id_col = resolve_identifier_col(df, args.id_col)
    if id_col == "__row_id__":
        df = df.copy()
        df[id_col] = df.index.astype(str)

    df = df.copy()
    df["__sort_id__"] = df[id_col].astype(str)

    out_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (input_path.parent / f"weight_sensitivity_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    ranked_tables: Dict[str, pd.DataFrame] = {}
    long_rows: List[pd.DataFrame] = []

    logger.info(f"Loaded input: {input_path}")
    logger.info(f"Identifier column: {id_col}")
    logger.info(f"Output directory: {out_dir}")
    top_ns = parse_top_ns(args.top_ns)
    logger.info(f"Top-N cutoffs: {', '.join(str(n) for n in top_ns)}")

    for scheme_name, weights in WEIGHT_SCHEMES:
        score_cols = ordered_unique([id_col, "__sort_id__", *DEFAULT_REQUIRED_COLUMNS])
        ranked = build_weighted_scores(df[score_cols].copy(), scheme_name, weights)
        ranked[id_col] = ranked[id_col].astype(str)
        ranked["compound_id"] = ranked[id_col]
        ranked["weight_qsar_prob"] = float(weights["qsar_prob"])
        ranked["weight_ad_score"] = float(weights["ad_score"])
        ranked["weight_admet_score"] = float(weights["admet_score"])
        ranked["scheme_label"] = scheme_name
        ranked_tables[scheme_name] = ranked
        long_rows.append(
            ranked[
                [
                    id_col,
                    "compound_id",
                    "scheme_label",
                    "weight_qsar_prob",
                    "weight_ad_score",
                    "weight_admet_score",
                    "final_score_raw",
                    "final_score",
                    "rank",
                ]
            ].copy()
        )

    ranking_df = pd.concat(long_rows, ignore_index=True)
    ranking_df.to_csv(out_dir / "weight_sensitivity_rankings.csv", index=False)

    summary_bundle = {
        "timestamp": datetime.now().isoformat(),
        "input_parquet": str(input_path),
        "output_dir": str(out_dir),
        "id_col": id_col,
        "top_ns": top_ns,
        "schemes": [{"name": scheme_name, "weights": weights} for scheme_name, weights in WEIGHT_SCHEMES],
        "n_input_rows": int(len(df)),
        "n_unique_ids": int(df[id_col].nunique()),
        "summaries": {},
    }
    matrices_by_top_n: Dict[int, pd.DataFrame] = {}

    for top_n in top_ns:
        top_sets = {
            scheme_name: set(table.loc[table["rank"] <= top_n, id_col].astype(str))
            for scheme_name, table in ranked_tables.items()
        }
        pairwise_df = overlap_summary(top_sets, top_n)
        pairwise_path = out_dir / f"weight_sensitivity_pairwise_overlap_top{top_n}.csv"
        pairwise_df.to_csv(pairwise_path, index=False)

        matrix_df = overlap_matrix(top_sets, top_n)
        matrix_path = out_dir / f"weight_sensitivity_overlap_matrix_top{top_n}.csv"
        matrix_df.to_csv(matrix_path)
        matrices_by_top_n[top_n] = matrix_df

        membership_df = top_membership_table(ranked_tables, top_sets, id_col)
        membership_path = out_dir / f"weight_sensitivity_top_membership_top{top_n}.csv"
        membership_df.to_csv(membership_path, index=False)

        min_pairwise_overlap = float(pairwise_df["overlap_ratio"].min()) if not pairwise_df.empty else 1.0
        intersection_ids = sorted(set.intersection(*top_sets.values())) if top_sets else []
        union_ids = sorted(set.union(*top_sets.values())) if top_sets else []
        statement = (
            f"Top {top_n} compounds showed >85% overlap across weighting schemes."
            if min_pairwise_overlap >= 0.85
            else f"Top {top_n} compounds showed a minimum pairwise overlap of {min_pairwise_overlap:.1%} across weighting schemes."
        )

        summary = {
            "timestamp": datetime.now().isoformat(),
            "input_parquet": str(input_path),
            "output_dir": str(out_dir),
            "id_col": id_col,
            "top_n": int(top_n),
            "schemes": [{"name": scheme_name, "weights": weights} for scheme_name, weights in WEIGHT_SCHEMES],
            "n_input_rows": int(len(df)),
            "n_unique_ids": int(df[id_col].nunique()),
            "pairwise_min_overlap_ratio": min_pairwise_overlap,
            "pairwise_min_overlap_percent": float(min_pairwise_overlap * 100.0),
            "intersection_size": int(len(intersection_ids)),
            "union_size": int(len(union_ids)),
            "statement": statement,
        }
        summary_path = out_dir / f"weight_sensitivity_summary_top{top_n}.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summary_bundle["summaries"][str(top_n)] = summary

        if args.make_plots:
            png_path, svg_path = plot_overlap_heatmap(matrix_df, out_dir, top_n)
            png_target = out_dir / f"weight_sensitivity_overlap_heatmap_top{top_n}.png"
            svg_target = out_dir / f"weight_sensitivity_overlap_heatmap_top{top_n}.svg"
            png_path.rename(png_target)
            svg_path.rename(svg_target)
            logger.info(f"Saved figure: {png_target}")
            logger.info(f"Saved figure: {svg_target}")

        logger.info(f"Saved pairwise overlap table: {pairwise_path}")
        logger.info(f"Saved overlap matrix: {matrix_path}")
        logger.info(f"Saved membership table: {membership_path}")
        logger.info(f"Saved summary: {summary_path}")
        logger.info(statement)

    if args.make_plots and len(matrices_by_top_n) > 1:
        combined_png, combined_svg = plot_combined_overlap_heatmaps(matrices_by_top_n, out_dir)
        logger.info(f"Saved combined figure: {combined_png}")
        logger.info(f"Saved combined figure: {combined_svg}")

    combined_summary_path = out_dir / "weight_sensitivity_summary.json"
    combined_summary_path.write_text(json.dumps(summary_bundle, indent=2), encoding="utf-8")
    logger.info(f"Saved rankings: {out_dir / 'weight_sensitivity_rankings.csv'}")
    logger.info(f"Saved combined summary: {combined_summary_path}")


if __name__ == "__main__":
    main()
