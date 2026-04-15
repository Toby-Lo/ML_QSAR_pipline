#!/usr/bin/env python3
"""
Optimized AD-Aware Virtual Screening Ranking Script

This script fuses calibrated probabilities with Applicability Domain (AD) scores
to prioritize compounds for docking. Designed for high-performance processing

Key optimizations:
- Single input file containing both predictions and AD scores
- Uses Pandas for efficient processing
- Implements 'Probability Space Shrinkage' logic validated in step22
- Supports both top-N and top-percent selection strategies
- Comprehensive logging with timestamps and progress bars
- Automatic timestamp-based output directory creation

Usage:
python scripts/step34_ad_aware_filter.py \
  --input-file models_out/qsar_ml_20260412_162829/virtual_screening/zinc_predictions_20260414_211051.parquet \
  --top-percent 0.1 \
  --ad-power 2.0 \
  --ad-threshold 0.30

python scripts/step34_ad_aware_filter.py \
  --input-file ./models_out/qsar_ml_20260412_162829/virtual_screening/zinc_predictions_20260414_211051.parquet \
  --top-n 10000 \
  --ad-power 2.0 \
  --ad-threshold 0.3 \
  --output-dir ./custom_output_dir
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging with both console and file handlers."""
    logger = logging.getLogger("ad_screening")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"screening_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    
    return logger


def load_merged_data(input_file: Path, logger: logging.Logger) -> pd.DataFrame:
    """
    Load merged predictions and AD scores from a single parquet file.
    
    Args:
        input_file: Path to the merged parquet file containing both predictions and AD scores
        logger: Logger instance for progress tracking
        
    Returns:
        Pandas DataFrame with required columns
        
    Raises:
        ValueError: If required columns are missing or data is invalid
    """
    logger.info(f"Loading merged predictions and AD scores from: {input_file}")
    
    # Get file size for progress estimation
    file_size_mb = input_file.stat().st_size / (1024 * 1024)
    logger.info(f"File size: {file_size_mb:.1f} MB")
    
    # Read the parquet file
    with tqdm(total=100, desc="Loading data", unit="%") as pbar:
        df = pd.read_parquet(input_file)
        pbar.update(100)
    
    # Validate required columns
    required_cols = {"zinc_id", "prob", "AD_Score"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in input file: {missing_cols}")
    
    # Validate data types and ranges
    prob_min, prob_max = df["prob"].min(), df["prob"].max()
    if prob_min < 0 or prob_max > 1:
        logger.warning(
            f"Probability values outside [0, 1] range detected: [{prob_min:.4f}, {prob_max:.4f}]. "
            "Consider checking calibration."
        )
    
    ad_min, ad_max = df["AD_Score"].min(), df["AD_Score"].max()
    if ad_min < 0 or ad_max > 1:
        logger.warning(
            f"AD Score values outside [0, 1] range detected: [{ad_min:.4f}, {ad_max:.4f}]"
        )
    
    logger.info(f"Loaded {len(df):,} records with both predictions and AD scores")
    return df


def apply_ad_shrinkage(prob: float, ad_score: float, ad_power: float) -> float:
    """
    Apply AD-aware probability shrinkage.
    
    Args:
        prob: Original probability score
        ad_score: Applicability Domain score
        ad_power: Power parameter for AD score scaling
        
    Returns:
        AD-aware fused score
    """
    if pd.isna(ad_score) or ad_score <= 0:
        return 0.0
    
    return prob * (ad_score ** ad_power)


def filter_and_score_data(df: pd.DataFrame, ad_power: float, ad_threshold: float, 
                         use_shrinkage: bool, logger: logging.Logger) -> pd.DataFrame:
    """
    Apply AD threshold filtering and calculate final scores.
    
    Args:
        df: Input DataFrame with predictions and AD scores
        ad_power: Power parameter for AD score scaling
        ad_threshold: Minimum AD score threshold
        use_shrinkage: Whether to apply AD shrinkage
        logger: Logger instance for progress tracking
        
    Returns:
        Processed DataFrame with final scores
    """
    logger.info("Applying AD threshold filtering and scoring...")
    
    # Apply AD threshold filtering
    initial_count = len(df)
    df_filtered = df[df["AD_Score"] >= ad_threshold].copy()
    filtered_count = len(df_filtered)
    
    retention_rate = filtered_count / initial_count if initial_count > 0 else 0
    logger.info(
        f"AD threshold filtering: {initial_count:,} → {filtered_count:,} "
        f"({retention_rate:.1%} retention)"
    )
    
    # Calculate final scores
    with tqdm(total=100, desc="Calculating scores", unit="%") as pbar:
        if use_shrinkage:
            df_filtered["Final_Score"] = df_filtered.apply(
                lambda row: apply_ad_shrinkage(row["prob"], row["AD_Score"], ad_power), 
                axis=1
            )
        else:
            # Fallback: use probability only
            df_filtered["Final_Score"] = df_filtered["prob"]
        pbar.update(100)
    
    return df_filtered


def rank_and_select_candidates(df: pd.DataFrame, top_n: Optional[int], 
                              top_percent: Optional[float], logger: logging.Logger) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rank candidates by final score and select top candidates.
    
    Args:
        df: DataFrame with final scores
        top_n: Number of top candidates to select
        top_percent: Percentage of top candidates to select
        logger: Logger instance for progress tracking
        
    Returns:
        Tuple of (ranked_df, top_candidates)
    """
    logger.info("Ranking candidates by final score...")
    
    # Sort by final score (descending)
    ranked_df = df.sort_values("Final_Score", ascending=False).reset_index(drop=True)
    
    # Select top candidates
    if top_n is not None:
        top_candidates = ranked_df.head(top_n)
        logger.info(f"Selected top {top_n} candidates")
    elif top_percent is not None:
        n_candidates = max(1, int(len(ranked_df) * top_percent))
        top_candidates = ranked_df.head(n_candidates)
        logger.info(f"Selected top {top_percent:.1%} ({n_candidates:,}) candidates")
    else:
        top_candidates = ranked_df
        logger.info("No selection criteria specified, returning all candidates")
    
    return ranked_df, top_candidates


def generate_summary(ranked_df: pd.DataFrame, top_candidates: pd.DataFrame,
                    params: Dict, logger: logging.Logger) -> Dict:
    """
    Generate screening summary.
    
    Args:
        ranked_df: Full ranked DataFrame
        top_candidates: Selected top candidates
        params: Parameter dictionary
        logger: Logger instance for progress tracking
        
    Returns:
        Summary dictionary
    """
    logger.info("Generating screening summary...")
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "parameters": params,
        "total_molecules": len(ranked_df),
        "candidates_selected": len(top_candidates),
    }
    
    if len(top_candidates) > 0:
        top_100 = top_candidates.head(100)
        
        summary.update({
            "top_100_mean_final_score": float(top_100["Final_Score"].mean()),
            "top_100_mean_prob": float(top_100["prob"].mean()),
            "top_100_mean_ad_score": float(top_100["AD_Score"].mean()),
            "overall_mean_final_score": float(ranked_df["Final_Score"].mean()),
            "overall_mean_prob": float(ranked_df["prob"].mean()),
            "overall_mean_ad_score": float(ranked_df["AD_Score"].mean()),
            })
    
    return summary


def save_outputs(ranked_df: pd.DataFrame, top_candidates: pd.DataFrame,
                summary: Dict, output_dir: Path, logger: logging.Logger) -> None:
    """
    Save all outputs to the specified directory.
    
    Args:
        ranked_df: Full ranked DataFrame
        top_candidates: Selected top candidates
        summary: Summary dictionary
        output_dir: Output directory
        logger: Logger instance for progress tracking
    """
    logger.info(f"Saving outputs to: {output_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save ranked results
    ranked_path = output_dir / "ranked_results.parquet"
    ranked_df.to_parquet(ranked_path, index=False)
    logger.info(f"Saved ranked results to: {ranked_path}")
    
    # Save top candidates
    top_candidates_path = output_dir / "top_candidates.parquet"
    top_candidates.to_parquet(top_candidates_path, index=False)
    logger.info(f"Saved top candidates to: {top_candidates_path}")
    
    # Save summary
    summary_path = output_dir / "screening_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to: {summary_path}")


def print_console_summary(summary: Dict, logger: logging.Logger) -> None:
    """
    Print a clean summary to the console.
    """
    logger.info("=" * 60)
    logger.info("SCREENING SUMMARY")
    logger.info("=" * 60)
    
    logger.info(f"Total molecules processed: {summary['total_molecules']:,}")
    logger.info(f"Candidates selected: {summary['candidates_selected']:,}")
    
    if 'top_100_mean_final_score' in summary:
        logger.info(f"Top 100 mean final score: {summary['top_100_mean_final_score']:.4f}")
        logger.info(f"Top 100 mean probability: {summary['top_100_mean_prob']:.4f}")
        logger.info(f"Top 100 mean AD score: {summary['top_100_mean_ad_score']:.4f}")
        logger.info(f"Overall mean final score: {summary['overall_mean_final_score']:.4f}")
    
    logger.info("=" * 60)


def main() -> None:
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="AD-aware virtual screening ranking with optimized single-file input"
    )
    
    # Required arguments
    parser.add_argument("--input-file", type=Path, required=True,
                       help="Path to merged parquet file containing both predictions and AD scores")
    
    # Selection strategy (mutually exclusive)
    selection_group = parser.add_mutually_exclusive_group()
    selection_group.add_argument("--top-n", type=int, default=10000,
                                help="Number of top candidates to select (default: 10000)")
    selection_group.add_argument("--top-percent", type=float,
                                help="Percentage of top candidates to select (e.g., 0.1 for top 0.1%)")
    
    # Optional parameters with defaults
    parser.add_argument("--ad-power", type=float, default=2.0,
                       help="Power parameter for AD score scaling (default: 2.0)")
    parser.add_argument("--ad-threshold", type=float, default=0.3,
                       help="Minimum AD score threshold (default: 0.3)")
    parser.add_argument("--use-shrinkage", action=argparse.BooleanOptionalAction, default=True,
                       help="Apply AD shrinkage (default: True)")
    parser.add_argument("--output-dir", type=Path,
                       help="Output directory (if not specified, creates timestamped directory)")
    
    args = parser.parse_args()
    
    # Validate file existence
    if not args.input_file.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
    
    # Determine output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        parent_dir = args.input_file.parent     # same level as the input path by defualt
        args.output_dir = parent_dir / f"ad_screening_results_{timestamp}"
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    
    # Parameter dictionary for summary
    params = {
        "input_file": str(args.input_file),
        "ad_power": args.ad_power,
        "top_n": args.top_n,
        "top_percent": args.top_percent,
        "ad_threshold": args.ad_threshold,
        "use_shrinkage": args.use_shrinkage,
        "output_dir": str(args.output_dir),
    }
    
    try:
        logger.info("Starting optimized AD-aware virtual screening ranking...")
        logger.info(f"Input file: {args.input_file}")
        logger.info(f"Output directory: {args.output_dir}")
        
        # Load data
        df = load_merged_data(args.input_file, logger)
        
        # Filter and score
        processed_df = filter_and_score_data(df, args.ad_power, args.ad_threshold, 
                                           args.use_shrinkage, logger)
        
        # Rank and select top candidates
        ranked_df, top_candidates = rank_and_select_candidates(
            processed_df, args.top_n, args.top_percent, logger
        )
        
        # Generate summary
        summary = generate_summary(ranked_df, top_candidates, params, logger)
        
        # Save outputs
        save_outputs(ranked_df, top_candidates, summary, args.output_dir, logger)
        
        # Print console summary
        print_console_summary(summary, logger)
        
        logger.info("Optimized AD-aware screening ranking completed successfully!")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()


# %%
# Plotting-only cell (interactive)
##############################################################################
# Goal:
#   - Visualize AD-aware screening results WITHOUT recomputation
#   - Diagnose AD impact on ranking and analyze chemical space coverage
#   - Export publication-ready figures (PNG + SVG)
##############################################################################

import os, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

def _in_ipython():
    try: return __IPYTHON__ is not None
    except NameError: return False

if _in_ipython():
    # --- USER EDITABLE ---
    OUT_DIR = Path("../models_out/qsar_ml_20260412_162829/virtual_screening/ad_screening_results_20260415_105720")
    # ---------------------

    PLOT_STYLE = {
        "font.family": "serif",
        "font.serif": ["Cambria"], # "Times New Roman", "Cambria"
        "font.size": 11,
        "axes.linewidth": 1.1,
        "grid.alpha": 0.25,
        "savefig.dpi": 600,
        "axes.spines.top": False,
        "axes.spines.right": False
    }

    def _setup():
        plt.rcParams.update(PLOT_STYLE)
        fig_dir = OUT_DIR / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        return fig_dir

    def _save(fig, name, fig_dir):
        for ext in ['png', 'svg']:
            fig.savefig(fig_dir / f"{name}.{ext}", bbox_inches='tight')
        plt.show()

    def _load():
        r_path, t_path = OUT_DIR / "ranked_results.parquet", OUT_DIR / "top_candidates.parquet"
        if not (r_path.exists() and t_path.exists()):
            raise FileNotFoundError(f"Missing data in {OUT_DIR}")
        return pd.read_parquet(r_path), pd.read_parquet(t_path)

    fig_dir = _setup()
    df, top_df = _load()

    # A. Distribution diagnostics
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col, color in zip(axes, ["prob", "AD_Score", "Final_Score"], ["#440154", "#21918c", "#fde725"]):
        sns.histplot(df[col], bins=50, kde=True, ax=ax, color=color, edgecolor='none')
        ax.set_title(f"Distribution: {col}")
    _save(fig, "fig1_distribution", fig_dir)

    # B. Relationship: Prob vs AD_Score
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(df["prob"], df["AD_Score"], c=df["Final_Score"], cmap="viridis", s=10, alpha=0.5)
    plt.colorbar(sc, label="Final_Score")
    ax.set_xlabel("Probability (Prob)")
    ax.set_ylabel("AD_Score")
    ax.set_title("AD-Aware Scoring Landscape")
    _save(fig, "fig2_prob_vs_ad", fig_dir)

    # C. Ranking Behavior
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df.index, df["Final_Score"], lw=2, color='black')
    ax.set_xlabel("Rank")
    ax.set_ylabel("Final_Score")
    ax.set_title("Screening Ranking Curve")
    _save(fig, "fig3_ranking_curve", fig_dir)

    # D. AD Impact (Overlap Analysis)
    top_n_list = [100, 500, 1000]
    overlap_ratios = []
    prob_ranked = df.sort_values("prob", ascending=False).index.tolist()
    final_ranked = df.index.tolist()
    
    for n in top_n_list:
        n = min(n, len(df))
        overlap = len(set(prob_ranked[:n]) & set(final_ranked[:n]))
        overlap_ratios.append(overlap / n)
        print(f"Top-{n} Overlap (Prob vs Final): {overlap/n:.1%}")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar([f"Top-{n}" for n in top_n_list], overlap_ratios, color="#3b528b")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Overlap Ratio")
    ax.set_title("Ranking Shift: Prob vs Final_Score")
    _save(fig, "fig4_overlap", fig_dir)

    # E. Top Candidate Bias
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col in zip(axes, ["prob", "AD_Score"]):
        sns.kdeplot(df[col], ax=ax, label="All", fill=True, alpha=0.3)
        sns.kdeplot(top_df[col], ax=ax, label="Top Hits", fill=True, alpha=0.3)
        ax.legend()
        ax.set_title(f"Selection Bias: {col}")
    _save(fig, "fig5_top_bias", fig_dir)

    # F. Chemical Space (PCA)
    features = [c for c in df.columns if c.startswith('morgan_') or c in ['AD_Score', 'prob']]
    if len(features) > 10:
        pca = PCA(n_components=2)
        X_all = pca.fit_transform(df[features].fillna(0))
        X_top = X_all[df.index.isin(top_df.index)]
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(X_all[:,0], X_all[:,1], c='lightgray', s=5, alpha=0.3, label="Screened")
        ax.scatter(X_top[:,0], X_top[:,1], c='#fde725', s=15, alpha=0.8, label="Top Hits", edgecolors='black', lw=0.5)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.legend()
        _save(fig, "fig6_pca_space", fig_dir)

    print(f"Success: All plots exported to {fig_dir}")
# %%
