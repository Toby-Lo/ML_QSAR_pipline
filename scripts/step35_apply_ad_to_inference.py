#!/usr/bin/env python3
"""
Production-grade AD-Aware Virtual Screening Ranking Script

This script fuses calibrated probabilities with Applicability Domain (AD) scores
to prioritize compounds for docking. Designed for high-performance processing

Key optimizations:
- Uses Polars for memory-efficient processing
- Avoids loading SMILES during merging/scoring phase
- Implements 'Probability Space Shrinkage' logic validated in step22
- Supports both top-N and top-percent selection strategies
- Comprehensive logging with timestamps and progress bars

Usage:
python scripts/step35_apply_ad_to_inference.py \
  --vs-predictions models_out/qsar_ml_20260412_162829/virtual_screening/zinc_predictions_20260412_171703.parquet \
  --ad-file models_out/qsar_ml_20260412_162829/split_seed_12345/validation/applicability_domain/SVC/seed_12345/ad_external_predictions.csv \
  --top-n 10000 \
  --ad-power 2.0 \
  --ad-threshold 0.3 \
  --output-dir models_out/qsar_ml_20260412_162829/virtual_screening/zinc_top_10k

python scripts/step35_apply_ad_to_inference.py \
  --vs-predictions ./models_out/qsar_ml_20260412_162829/virtual_screening/zinc_predictions_20260412_171703.parquet \
  --ad-file ./models_out/qsar_ml_20260412_162829/split_seed_12345/validation/applicability_domain/SVC/seed_12345/ad_external_predictions.csv \
  --top-percent 0.1 \
  --ad-power 2.0 \
  --ad-threshold 0.3 \
  --output-dir ./models_out/qsar_ml_20260412_162829/virtual_screening/zinc_top_0.1pct
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
import polars as pl
from tqdm import tqdm


def setup_logging(output_dir: Path) -> logging.Logger:
    """
    Setup comprehensive logging with timestamps and file output.
    
    Args:
        output_dir: Directory for log files
        
    Returns:
        Configured logger instance
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"screening_{timestamp}.log"
    
    # Clear existing handlers
    logging.getLogger().handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)-8s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


def _require_deps() -> None:
    """Ensure required dependencies are available."""
    try:
        import numpy as _np  # noqa: F401
        import polars as _pl  # noqa: F401
        import tqdm as _tqdm  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing runtime dependencies. Please install: numpy, polars, tqdm.\n"
            f"Import error: {exc}"
        ) from exc


def apply_ad_shrinkage(prob: pl.Expr, ad_score: pl.Expr, power: float = 2.0) -> pl.Expr:
    """
    Apply AD shrinkage: Final_Score = prob * (AD_Score^power)
    
    This implements the 'Probability Space Shrinkage' logic validated in step22.
    
    Args:
        prob: Probability expression (float32)
        ad_score: AD_Score expression (float32)
        power: Power parameter for AD score scaling
        
    Returns:
        Expression for final score calculation
    """
    return prob * (ad_score.pow(power))


def load_vs_predictions(vs_path: Path, logger: logging.Logger) -> pl.DataFrame:
    """
    Load virtual screening predictions with validation and progress tracking.
    
    Args:
        vs_path: Path to VS predictions parquet file
        logger: Logger instance for progress tracking
        
    Returns:
        Polars DataFrame with required columns
        
    Raises:
        ValueError: If required columns are missing or data is invalid
    """
    logger.info(f"Loading VS predictions from: {vs_path}")
    
    # Get file size for progress estimation
    file_size_mb = vs_path.stat().st_size / (1024 * 1024)
    logger.info(f"File size: {file_size_mb:.1f} MB")
    
    # Read with minimal columns to save memory
    with tqdm(total=100, desc="Loading VS predictions", unit="%") as pbar:
        df = pl.read_parquet(
            vs_path,
            columns=["zinc_id", "prob", "pred_label"]
        )
        pbar.update(100)
    
    # Validate required columns
    required_cols = {"zinc_id", "prob", "pred_label"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in VS predictions: {missing_cols}")
    
    # Validate data types and ranges
    prob_min, prob_max = df["prob"].min(), df["prob"].max()
    if prob_min < 0 or prob_max > 1:
        logger.warning(
            f"Probability values outside [0, 1] range detected: [{prob_min:.4f}, {prob_max:.4f}]. "
            "Consider checking calibration."
        )
    
    logger.info(f"Loaded {len(df):,} VS predictions")
    return df


def load_ad_results(ad_path: Path, ad_threshold: float, logger: logging.Logger) -> pl.DataFrame:
    """
    Load AD results with threshold filtering and progress tracking.
    Supports both 'id' (from step22) and 'zinc_id' (from VS inference) column names.
    """
    logger.info(f"Loading AD results from: {ad_path}")
    
    # 获取文件大小
    file_size_mb = ad_path.stat().st_size / (1024 * 1024)
    logger.info(f"File size: {file_size_mb:.1f} MB")
    
    with tqdm(total=100, desc="Loading AD results", unit="%") as pbar:
        # 1. 先不限制 columns 读入，或者读入所有候选列名
        df = pl.read_csv(ad_path)
        
        # 2. 核心逻辑：自动映射 id -> zinc_id
        if "id" in df.columns and "zinc_id" not in df.columns:
            df = df.rename({"id": "zinc_id"})
            logger.info("Automatically mapped column 'id' to 'zinc_id'")

        try:
            df = df.with_columns(
                pl.col("zinc_id").cast(pl.Int64, strict=False)
            )
            # 剔除无法转换 ID 的行（例如标题行重复或包含非数字字符）
            df = df.filter(pl.col("zinc_id").is_not_null())
        except Exception as e:
            logger.error(f"Failed to cast 'zinc_id' to Int64: {e}")
            raise
        
        # 3. 确保必要的列存在并精简 DataFrame 节省内存
        required_cols = ["zinc_id", "AD_Score"]
        available_cols = [c for c in required_cols if c in df.columns]
        
        if len(available_cols) < 2:
            missing = set(required_cols) - set(df.columns)
            raise ValueError(f"Missing required columns in AD results: {missing}. Found: {df.columns}")
            
        df = df.select(required_cols)
        pbar.update(100)
    
    # 4. 应用阈值过滤
    initial_count = len(df)
    df = df.filter(pl.col("AD_Score") >= ad_threshold)
    filtered_count = len(df)
    
    retention_rate = filtered_count / initial_count if initial_count > 0 else 0
    logger.info(
        f"AD threshold filtering: {initial_count:,} → {filtered_count:,} "
        f"({retention_rate:.1%} retention)"
    )
    
    return df


def merge_and_score(
    vs_df: pl.DataFrame,
    ad_df: pl.DataFrame,
    ad_power: float,
    use_shrinkage: bool,
    logger: logging.Logger
) -> pl.DataFrame:
    """
    Merge VS predictions with AD results and calculate final scores.
    
    Args:
        vs_df: VS predictions DataFrame
        ad_df: AD results DataFrame
        ad_power: Power parameter for AD scoring
        use_shrinkage: Whether to apply AD shrinkage
        logger: Logger instance for progress tracking
        
    Returns:
        Merged DataFrame with final scores
    """
    logger.info("Merging VS predictions with AD results...")
    
    with tqdm(total=3, desc="Merging and scoring", unit="steps") as pbar:
        # Perform left join on zinc_id
        merged_df = vs_df.join(ad_df, on="zinc_id", how="left")
        pbar.update(1)
        
        # Check merge success rate
        total_vs = len(vs_df)
        matched_count = merged_df["AD_Score"].is_not_null().sum()
        match_rate = matched_count / total_vs
        
        logger.info(
            f"Merge success: {matched_count:,}/{total_vs:,} "
            f"({match_rate:.1%}) compounds matched"
        )
        
        if match_rate < 0.9:
            logger.warning(
                f"Low merge success rate ({match_rate:.1%}). "
                "Check zinc_id alignment between files."
            )
        pbar.update(1)
        
        # Calculate final score
        if use_shrinkage:
            merged_df = merged_df.with_columns([
                apply_ad_shrinkage(
                    pl.col("prob"),
                    pl.col("AD_Score"),
                    ad_power
                ).alias("Final_Score")
            ])
        else:
            # Fallback: use probability only
            merged_df = merged_df.with_columns([
                pl.col("prob").alias("Final_Score")
            ])
        pbar.update(1)
    
    return merged_df


def calculate_selection_count(
    total_molecules: int,
    top_n: Optional[int],
    top_percent: Optional[float]
) -> int:
    """
    Calculate the number of molecules to select based on top-N and top-percent criteria.
    
    Args:
        total_molecules: Total number of molecules
        top_n: Fixed number of molecules to select
        top_percent: Percentage of molecules to select
        
    Returns:
        Number of molecules to select
        
    Raises:
        ValueError: If both top_n and top_percent are None
    """
    if top_n is not None and top_percent is not None:
        raise ValueError("Cannot specify both --top-n and --top-percent")
    
    if top_n is not None:
        return min(top_n, total_molecules)
    
    if top_percent is not None:
        if not 0 < top_percent <= 100:
            raise ValueError("top_percent must be between 0 and 100")
        count = int(total_molecules * (top_percent / 100))
        return max(1, count)  # At least 1 molecule
    
    raise ValueError("Must specify either --top-n or --top-percent")


def rank_and_select_candidates(
    merged_df: pl.DataFrame,
    top_n: Optional[int],
    top_percent: Optional[float],
    vs_path: Path,
    logger: logging.Logger
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Rank compounds by final score and select top candidates using flexible selection strategy.
    
    Args:
        merged_df: Merged DataFrame with scores
        top_n: Number of top candidates to select (None if using top_percent)
        top_percent: Percentage of top candidates to select (None if using top_n)
        vs_path: Path to original VS predictions for SMILES retrieval
        logger: Logger instance for progress tracking
        
    Returns:
        Tuple of (full ranked DataFrame, top candidates DataFrame)
    """
    total_molecules = len(merged_df)
    selection_count = calculate_selection_count(total_molecules, top_n, top_percent)
    
    # Log selection strategy
    if top_n is not None:
        logger.info(f"Ranking compounds and selecting top-{top_n} candidates...")
    else:
        logger.info(f"Ranking compounds and selecting top {top_percent}% candidates...")
    
    logger.info(f"Selection count: {selection_count:,} out of {total_molecules:,}")
    
    with tqdm(total=3, desc="Ranking and selection", unit="steps") as pbar:
        # Sort by final score descending
        ranked_df = merged_df.sort("Final_Score", descending=True)
        pbar.update(1)
        
        # Get zinc_ids of top candidates
        top_zinc_ids = ranked_df.head(selection_count).select("zinc_id").to_series()
        pbar.update(1)
        
        # Load SMILES only for top candidates to save memory
        logger.info("Loading SMILES for top candidates (using lazy scan)...")
        smiles_df = (
            pl.scan_parquet(vs_path)
            .select(["zinc_id", "smiles"])
            .filter(pl.col("zinc_id").is_in(top_zinc_ids))
            .collect() # 只有在这里才会真正执行读取和过滤
        )
        
        # Join SMILES back to top candidates
        top_candidates = ranked_df.head(selection_count).join(smiles_df, on="zinc_id", how="left")
        
        # Reorder columns for final output
        final_columns = ["zinc_id", "smiles", "prob", "AD_Score", "Final_Score", "pred_label"]
        top_candidates = top_candidates.select(final_columns)
        pbar.update(1)
    
    return ranked_df, top_candidates


def generate_summary(
    ranked_df: pl.DataFrame,
    top_candidates: pl.DataFrame,
    params: Dict,
    logger: logging.Logger
) -> Dict:
    """
    生成筛选总结，能够区分有效 0.0 与数据缺失 (null)。
    """
    logger.info("Generating screening summary...")
    
    # 统计有多少行因为 AD 缺失导致 Final_Score 为 null
    null_count = ranked_df["Final_Score"].is_null().sum()
    if null_count > 0:
        logger.warning(f"Detected {null_count:,} molecules with null Final_Score (likely due to missing AD scores).")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "parameters": params,
        "total_molecules": len(ranked_df),
        "molecules_passing_ad": int(ranked_df["AD_Score"].is_not_null().sum()),
        "null_score_count": int(null_count),
        "candidates_selected": len(top_candidates),
    }

    def get_safe_mean(series: pl.Series) -> Optional[float]:
        # drop_nulls() 后计算均值，如果全是 null，mean() 会返回 None
        m = series.drop_nulls().mean()
        return float(m) if m is not None else None

    if len(top_candidates) > 0:
        top_100 = top_candidates.head(100)
        
        # 这里不使用 fill_null(0.0)，而是保留 None
        summary.update({
            "top_100_mean_final_score": get_safe_mean(top_100["Final_Score"]),
            "top_100_mean_prob": get_safe_mean(top_100["prob"]),
            "overall_mean_final_score": get_safe_mean(ranked_df["Final_Score"]),
            "overall_mean_ad_score": get_safe_mean(ranked_df["AD_Score"]),
        })
    
    return summary

def save_outputs(
    ranked_df: pl.DataFrame,
    top_candidates: pl.DataFrame,
    summary: Dict,
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """
    Save all output files in parquet and JSON formats.
    
    Args:
        ranked_df: Full ranked DataFrame
        top_candidates: Top candidates DataFrame
        summary: Screening summary dictionary
        output_dir: Output directory path
        logger: Logger instance for progress tracking
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with tqdm(total=3, desc="Saving outputs", unit="files") as pbar:
        # Save full ranked set (without SMILES to save space)
        ranked_df.select(["zinc_id", "prob", "AD_Score", "Final_Score", "pred_label"]).write_parquet(
            output_dir / "ranked_vs_ad_full.parquet",
            compression="zstd"
        )
        pbar.update(1)
        
        # Save top candidates with SMILES
        top_candidates.write_parquet(
            output_dir / "top_selected_candidates.parquet",
            compression="zstd"
        )
        pbar.update(1)
        
        # Save summary
        with open(output_dir / "screening_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        pbar.update(1)
    
    logger.info(f"Outputs saved to: {output_dir}")


def print_console_summary(summary: Dict, logger: logging.Logger) -> None:
    """
    Print a clean summary to the console with None-safe formatting.
    """
    def fmt(val, format_str=".4f"):
        # 如果值是 None，显示 N/A，否则按格式显示
        if val is None:
            return "N/A"
        return f"{val:{format_str}}"

    logger.info("\n" + "="*60)
    logger.info("SCREENING SUMMARY")
    logger.info("="*60)
    logger.info(f"Total molecules processed:    {summary['total_molecules']:,}")
    logger.info(f"Molecules passing AD:         {summary['molecules_passing_ad']:,}")
    logger.info(f"Null scores detected:         {summary['null_score_count']:,}")
    logger.info(f"Candidates selected:          {summary['candidates_selected']:,}")
    
    # 打印统计均值
    logger.info(f"\nTop 100 mean Final_Score:    {fmt(summary.get('top_100_mean_final_score'))}")
    logger.info(f"Top 100 mean Probability:    {fmt(summary.get('top_100_mean_prob'))}")
    logger.info(f"Overall mean AD Score:       {fmt(summary.get('overall_mean_ad_score'))}")
    logger.info("="*60 + "\n")

def main() -> None:
    """Main execution function."""
    _require_deps()
    
    parser = argparse.ArgumentParser(
        description="AD-aware virtual screening ranking with memory optimization"
    )
    
    # Required arguments
    parser.add_argument("--vs-predictions", type=Path, required=True,
                       help="Path to VS predictions parquet file")
    parser.add_argument("--ad-file", type=Path, required=True,
                       help="Path to AD results CSV file")
    
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
    parser.add_argument("--output-dir", type=Path, default=Path("./ad_screening_results"),
                       help="Output directory (default: ./ad_screening_results)")
    
    args = parser.parse_args()
    
    # Validate file existence
    if not args.vs_predictions.exists():
        raise FileNotFoundError(f"VS predictions file not found: {args.vs_predictions}")
    if not args.ad_file.exists():
        raise FileNotFoundError(f"AD file not found: {args.ad_file}")
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    
    # Parameter dictionary for summary
    params = {
        "ad_power": args.ad_power,
        "top_n": args.top_n,
        "top_percent": args.top_percent,
        "ad_threshold": args.ad_threshold,
        "use_shrinkage": args.use_shrinkage,
    }
    
    try:
        logger.info("Starting AD-aware virtual screening ranking...")
        logger.info(f"VS predictions: {args.vs_predictions}")
        logger.info(f"AD results: {args.ad_file}")
        
        # Load data
        vs_df = load_vs_predictions(args.vs_predictions, logger)
        ad_df = load_ad_results(args.ad_file, args.ad_threshold, logger)
        
        # Merge and score
        merged_df = merge_and_score(vs_df, ad_df, args.ad_power, args.use_shrinkage, logger)
        
        # Rank and select top candidates
        ranked_df, top_candidates = rank_and_select_candidates(
            merged_df, args.top_n, args.top_percent, args.vs_predictions, logger
        )
        
        # Generate summary
        summary = generate_summary(ranked_df, top_candidates, params, logger)
        
        # Save outputs
        save_outputs(ranked_df, top_candidates, summary, args.output_dir, logger)
        
        # Print console summary
        print_console_summary(summary, logger)
        
        logger.info("AD-aware screening ranking completed successfully!")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()