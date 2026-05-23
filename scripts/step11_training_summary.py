"""
Training Summary Aggregator - Compiles per-seed metrics from QSAR ML pipeline runs.

Purpose:
  Aggregate external test set or cross-validation metrics across multiple random seeds
  to identify which seed configuration is most representative (e.g., closest to mean).
  
Usage:
  python step11_training_summary.py --run-dir models_out/qsar_ml_20260410_124055
  python step11_training_summary.py --run-dir models_out/qsar_ml_20260410_124055 \\
                                    --output results/seed_metrics_summary.csv \\
                                    --metrics mcc,f1,accuracy \\
                                    --sort-by RFC_mcc \\
                                    --stage cv

python scripts/step11_training_summary.py --run-dir models_out/qsar_ml_20260412_162829 \
--stage cv --sort-metric RFC_mcc


"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import statistics


def setup_logger() -> logging.Logger:
    """Configure logger for console output."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


logger = setup_logger()


def find_seed_directories(run_dir: Path) -> List[int]:
    """Find all split_seed_N directories in the run directory.
    
    Args:
        run_dir: Path to qsar_ml_YYYYMMDD_HHMMSS directory
        
    Returns:
        Sorted list of seed integers
    """
    seeds = []
    for item in run_dir.iterdir():
        if item.is_dir() and item.name.startswith("split_seed_"):
            try:
                seed = int(item.name.replace("split_seed_", ""))
                seeds.append(seed)
            except ValueError:
                pass
    return sorted(seeds)


def parse_summary_file(csv_path: Path, stage: str) -> Dict[str, Dict[str, float]]:
    """Parse metrics summary CSV for a single seed.
    
    Args:
        csv_path: Path to the CSV file
        stage: 'external' or 'cv'
        
    Returns:
        Dict mapping model name -> dict of metrics
        Example: {'LR': {'mcc': 0.643, 'f1': 0.900, ...}, ...}
    """
    try:
        df = pd.read_csv(csv_path)
        results = {}
        
        if stage == "external":
            for _, row in df.iterrows():
                model = row['model']
                # Extract all numeric columns that are metric values
                metrics = {}
                for col in df.columns:
                    if col != 'model' and '_mean_std' not in col and '_std' not in col:
                        # _mean columns are the actual metric values
                        if col.endswith('_mean'):
                            metric_name = col.replace('_mean', '')
                            metrics[metric_name] = float(row[col])
                results[model] = metrics
        elif stage == "cv":
            for model, group in df.groupby('model'):
                metrics = {}
                for _, row in group.iterrows():
                    if not pd.isna(row['mean']):
                        metrics[row['metric']] = float(row['mean'])
                results[model] = metrics
        
        return results
    except Exception as e:
        logger.warning(f"Failed to parse {csv_path}: {e}")
        return {}


def aggregate_metrics(
    run_dir: Path,
    stage: str = "external",
    seeds: Optional[List[int]] = None
) -> Tuple[Dict[int, Dict], List[str], List[str]]:
    """Aggregate metrics from all seeds.
    
    Args:
        run_dir: Path to qsar_ml_YYYYMMDD_HHMMSS directory
        stage: 'external' or 'cv'
        seeds: Specific seeds to process (if None, find all)
        
    Returns:
        Tuple of (aggregated_data, model_names, metric_names)
        aggregated_data: {seed: {model: {metric: value}}}
    """
    if seeds is None:
        seeds = find_seed_directories(run_dir)
    
    if not seeds:
        raise ValueError(f"No seed directories found in {run_dir}")
    
    logger.info(f"Found {len(seeds)} seed(s): {seeds}")
    
    aggregated = {}
    all_models = set()
    all_metrics = set()
    
    for seed in seeds:
        seed_dir = run_dir / f"split_seed_{seed}"
        file_name = "external_test_summary.csv" if stage == "external" else "cv_summary.csv"
        results_file = seed_dir / "results" / file_name
        
        if not results_file.exists():
            logger.warning(f"Results file not found: {results_file}")
            continue
        
        logger.info(f"Processing seed {seed}...")
        metrics = parse_summary_file(results_file, stage)
        aggregated[seed] = metrics
        
        for model, model_metrics in metrics.items():
            all_models.add(model)
            all_metrics.update(model_metrics.keys())
    
    if not aggregated:
        raise ValueError("No data could be extracted from seed directories")
    
    return aggregated, sorted(list(all_models)), sorted(list(all_metrics))


def create_summary_table(
    aggregated: Dict[int, Dict],
    models: List[str],
    metrics: List[str],
    primary_metric: str = "mcc"
) -> pd.DataFrame:
    """Create a summary table with seeds as rows and model/metric combinations as columns.
    
    Args:
        aggregated: Aggregated metrics data
        models: List of model names
        metrics: List of metric names
        primary_metric: Metric to highlight for seed selection
        
    Returns:
        DataFrame with mean row first, then one row per seed
    """
    seeds = sorted(aggregated.keys())
    
    # Create column names: model_metric (e.g., "LR_mcc", "RFC_f1")
    columns = []
    for model in models:
        for metric in metrics:
            columns.append(f"{model}_{metric}")
    
    # Prepare data
    data_dict = {}
    for seed in seeds:
        row = []
        for model in models:
            for metric in metrics:
                if model in aggregated[seed] and metric in aggregated[seed][model]:
                    row.append(aggregated[seed][model][metric])
                else:
                    row.append(np.nan)
        data_dict[f"seed_{seed}"] = row
    
    # Create DataFrame with seeds as rows
    df = pd.DataFrame(data_dict, index=columns).T
    
    # Calculate mean and std
    mean_row = df.mean()
    std_row = df.std()
    
    # Add mean and std as new rows
    df.loc["MEAN"] = mean_row
    df.loc["STD"] = std_row
    
    # Reorder to put MEAN and STD at the top
    df = df.reindex(["MEAN", "STD"] + [f"seed_{s}" for s in seeds])
    
    return df


def calculate_distance_from_mean(
    df: pd.DataFrame,
    target_metric: Optional[str] = None
) -> Dict[int, float]:
    """Calculate distance from mean for each seed.
    
    For seeds, calculate Euclidean distance from mean across all metrics.
    Can focus on a single target_metric if specified.
    
    Args:
        df: Summary table (with MEAN as first row)
        target_metric: If specified, only use this metric for distance calculation
                      (format: "MODEL_METRIC" e.g., "RFC_mcc")
        
    Returns:
        Dict mapping seed -> distance
    """
    mean_row = df.loc["MEAN"]
    distances = {}
    
    for idx in df.index:
        if idx.startswith("seed_"):
            seed = int(idx.split("_")[1])
            row = df.loc[idx]
            
            if target_metric:
                # Single metric distance
                if target_metric in mean_row.index:
                    dist = abs(row[target_metric] - mean_row[target_metric])
                else:
                    logger.warning(f"Metric '{target_metric}' not found in columns")
                    dist = np.inf
            else:
                # Euclidean distance across all metrics
                valid_mask = ~(row.isna() | mean_row.isna())
                if not valid_mask.any():
                    dist = np.inf
                else:
                    dist = np.sqrt(np.sum((row[valid_mask] - mean_row[valid_mask]) ** 2))
            
            distances[seed] = dist
    
    return distances


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate training metrics across seeds to identify representative configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - summarize all models and metrics
  python step11_training_summary.py --run-dir models_out/qsar_ml_20260410_124055
  
  # Specify output file and metrics
  python step11_training_summary.py \\
    --run-dir models_out/qsar_ml_20260410_124055 \\
    --output results/seed_comparison.csv \\
    --metrics mcc,f1,accuracy
  
  # Sort by distance from mean for a specific metric
  python step11_training_summary.py \\
    --run-dir models_out/qsar_ml_20260410_124055 \\
    --sort-by mcc \\
    --sort-metric RFC_mcc
        """
    )
    
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to qsar_ml_YYYYMMDD_HHMMSS run directory"
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["external", "cv"],
        default="external",
        help="Which stage to summarize: external (default) or cv"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV file path. If not specified, saves to run_dir/results/seed_metrics_summary.csv"
    )
    parser.add_argument(
        "--write-across-seed-summary",
        action="store_true",
        default=True,
        help="Also write a long-form across-seed summary CSV (model,metric,mean,std,n,ddof) to run_dir/results/",
    )
    parser.add_argument(
        "--no-write-across-seed-summary",
        action="store_false",
        dest="write_across_seed_summary",
        help="Disable writing the extra across-seed summary CSV.",
    )
    parser.add_argument(
        "--across-seed-ddof",
        type=int,
        choices=[0, 1],
        default=1,
        help="ddof for across-seed std in the extra summary CSV (default: 1, sample std).",
    )
    parser.add_argument(
        "--write-final-tables",
        action="store_true",
        default=True,
        help="Also write manuscript-ready tables (mean ± std, 3 decimals) for CV and External to run_dir/results/final_table/.",
    )
    parser.add_argument(
        "--no-write-final-tables",
        action="store_false",
        dest="write_final_tables",
        help="Disable writing manuscript-ready tables to run_dir/results/final_table/.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="mcc,roc_auc,pr_auc,accuracy,precision,recall,f1",
        help="Comma-separated list of metrics to include (default: mcc,f1,accuracy,precision,recall,roc_auc)"
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of models to include. If not specified, includes all"
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="closest-to-mean",
        help="Sort seeds by: closest-to-mean (Euclidean), a general metric (e.g., mcc), or a specific model metric (e.g., RFC_mcc)"
    )
    parser.add_argument(
        "--sort-metric",
        type=str,
        default=None,
        help="For --sort-by closest-to-mean, use this metric for single-metric distance. Format: MODEL_METRIC (e.g., RFC_mcc)"
    )
    
    args = parser.parse_args()
    
    # Validation
    if not args.run_dir.exists():
        logger.error(f"Run directory not found: {args.run_dir}")
        sys.exit(1)
    
    # Parse metrics and models
    requested_metrics = [m.strip() for m in args.metrics.split(",")]
    requested_models = None
    if args.models:
        requested_models = [m.strip() for m in args.models.split(",")]
    
    try:
        # Aggregate data
        logger.info(f"Aggregating {args.stage} metrics from all seeds...")
        aggregated, all_models, all_metrics = aggregate_metrics(args.run_dir, stage=args.stage)
        
        # Filter to requested metrics/models
        available_metrics = [m for m in requested_metrics if m in all_metrics]
        if not available_metrics:
            logger.warning(
                f"None of requested metrics {requested_metrics} found. "
                f"Available: {all_metrics}"
            )
            available_metrics = all_metrics
        
        final_models = all_models
        if requested_models:
            final_models = [m for m in requested_models if m in all_models]
            if not final_models:
                final_models = all_models
        
        logger.info(f"Using {len(final_models)} models: {final_models}")
        logger.info(f"Using {len(available_metrics)} metrics: {available_metrics}")
        
        # Create summary table
        summary_df = create_summary_table(aggregated, final_models, available_metrics)
        
        # Determine sorting logic
        is_distance = False
        sort_values = {}
        if args.sort_by == "closest-to-mean":
            sort_values = calculate_distance_from_mean(summary_df, target_metric=args.sort_metric)
            sorted_seeds = sorted(sort_values, key=sort_values.get) if sort_values else []
            sort_desc = "closest-to-mean" + (f" ({args.sort_metric})" if args.sort_metric else "")
            is_distance = True
        else:
            if args.sort_by in summary_df.columns:
                for idx in summary_df.index:
                    if idx.startswith("seed_"):
                        seed = int(idx.split("_")[1])
                        sort_values[seed] = summary_df.loc[idx, args.sort_by]
                sorted_seeds = sorted(sort_values, key=sort_values.get, reverse=True) if sort_values else []
                sort_desc = f"highest {args.sort_by}"
            else:
                cols = [c for c in summary_df.columns if c.endswith(f"_{args.sort_by}")]
                if not cols:
                    logger.warning(f"Sort metric '{args.sort_by}' not found. Falling back to closest-to-mean.")
                    sort_values = calculate_distance_from_mean(summary_df)
                    sorted_seeds = sorted(sort_values, key=sort_values.get) if sort_values else []
                    sort_desc = "closest-to-mean (fallback)"
                    is_distance = True
                else:
                    for idx in summary_df.index:
                        if idx.startswith("seed_"):
                            seed = int(idx.split("_")[1])
                            sort_values[seed] = summary_df.loc[idx, cols].mean()
                    sorted_seeds = sorted(sort_values, key=sort_values.get, reverse=True) if sort_values else []
                    sort_desc = f"highest average {args.sort_by}"

        if sort_values:
            top_seed = sorted_seeds[0] if sorted_seeds else None
            logger.info(
                f"\n{'='*80}"
                f"\n  Seed Analysis (sorted by {sort_desc})"
                f"\n{'='*80}"
            )
            for seed in sorted_seeds:
                val = sort_values[seed]
                if is_distance:
                    marker = " ← CLOSEST TO MEAN" if seed == top_seed else ""
                    logger.info(f"  seed_{seed:5d} | distance: {val:.6f}{marker}")
                else:
                    marker = " ← BEST" if seed == top_seed else ""
                    logger.info(f"  seed_{seed:5d} | value: {val:.6f}{marker}")
        
        # Sort and reorder DataFrame
        final_order = ["MEAN", "STD"] + [f"seed_{s}" for s in sorted_seeds]
        summary_df = summary_df.loc[final_order]
        
        # Determine output path
        output_path = args.output
        if output_path is None:
            output_name = "seed_metrics_summary.csv" if args.stage == "external" else "seed_cv_metrics_summary.csv"
            output_path = args.run_dir / "results" / output_name
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        summary_df.to_csv(output_path)
        logger.info(f"\n✓ Summary saved to: {output_path}")

        if args.write_across_seed_summary:
            rows = []
            seed_ids = sorted(aggregated.keys())
            for model in final_models:
                for metric in available_metrics:
                    values = []
                    for seed in seed_ids:
                        v = aggregated.get(seed, {}).get(model, {}).get(metric)
                        if v is None:
                            continue
                        try:
                            fv = float(v)
                        except Exception:
                            continue
                        if np.isnan(fv):
                            continue
                        values.append(fv)
                    n = len(values)
                    if n == 0:
                        mean_val = None
                        std_val = None
                    else:
                        mean_val = float(statistics.fmean(values))
                        if n <= 1:
                            std_val = 0.0
                        else:
                            std_val = float(statistics.pstdev(values) if args.across_seed_ddof == 0 else statistics.stdev(values))
                    rows.append({
                        "model": model,
                        "metric": metric,
                        "mean": mean_val,
                        "std": std_val,
                        "n": n,
                        "ddof": args.across_seed_ddof,
                    })

            across_df = pd.DataFrame(rows)
            across_name = (
                "all_seed_cv_summary_across_seeds.csv"
                if args.stage == "cv"
                else "all_seed_external_summary_across_seeds.csv"
            )
            across_path = args.run_dir / "results" / across_name
            across_path.parent.mkdir(parents=True, exist_ok=True)
            across_df.to_csv(across_path, index=False)
            logger.info(f"✓ Across-seed summary saved to: {across_path}")

        if args.write_final_tables:
            def fmt_cell(mean_val, std_val) -> str:
                if mean_val is None or std_val is None:
                    return ""
                try:
                    if (isinstance(mean_val, float) and np.isnan(mean_val)) or (isinstance(std_val, float) and np.isnan(std_val)):
                        return ""
                except Exception:
                    pass
                return f"{float(mean_val):.3f} ± {float(std_val):.3f}"

            def build_final_table(stage: str) -> pd.DataFrame:
                aggregated2, all_models2, all_metrics2 = aggregate_metrics(args.run_dir, stage=stage)
                models2 = all_models2
                if requested_models:
                    models2 = [m for m in requested_models if m in all_models2] or all_models2
                metrics2 = [m for m in requested_metrics if m in all_metrics2] or all_metrics2

                seed_ids = sorted(aggregated2.keys())
                rows = []
                for model in models2:
                    row = {"model": model}
                    for metric in metrics2:
                        values = []
                        for seed in seed_ids:
                            v = aggregated2.get(seed, {}).get(model, {}).get(metric)
                            if v is None:
                                continue
                            try:
                                fv = float(v)
                            except Exception:
                                continue
                            if np.isnan(fv):
                                continue
                            values.append(fv)
                        if not values:
                            row[metric] = ""
                            continue
                        mean_val = float(statistics.fmean(values))
                        if len(values) <= 1:
                            std_val = 0.0
                        else:
                            std_val = float(
                                statistics.pstdev(values)
                                if args.across_seed_ddof == 0
                                else statistics.stdev(values)
                            )
                        row[metric] = fmt_cell(mean_val, std_val)
                    rows.append(row)
                df = pd.DataFrame(rows)
                if not df.empty:
                    # Keep "model" first, then metrics order
                    col_order = ["model"] + [m for m in metrics2 if m in df.columns]
                    df = df[col_order]
                return df

            final_dir = args.run_dir / "results" / "final_table"
            final_dir.mkdir(parents=True, exist_ok=True)
            cv_table = build_final_table("cv")
            ext_table = build_final_table("external")
            cv_path = final_dir / "final_table_cv_mean_std.csv"
            ext_path = final_dir / "final_table_external_mean_std.csv"
            cv_table.to_csv(cv_path, index=False)
            ext_table.to_csv(ext_path, index=False)
            logger.info(f"✓ Final table (CV) saved to: {cv_path}")
            logger.info(f"✓ Final table (External) saved to: {ext_path}")
        
        # Print summary
        logger.info(f"\nSummary Table:")
        logger.info(f"\n{summary_df.to_string()}")
        
        # Additional analysis: suggest seed for downstream analysis
        if sort_values and sorted_seeds:
            top_seed = sorted_seeds[0]
            val = sort_values[top_seed]
            if is_distance:
                rec_text = f"closest to mean, distance={val:.6f}"
            else:
                rec_text = f"best {args.sort_by}, value={val:.6f}"
            logger.info(
                f"\n{'='*80}"
                f"\nRECOMMENDATION FOR DOWNSTREAM ANALYSIS:"
                f"\n  Use seed_{top_seed} ({rec_text})"
                f"\n  Run: python scripts/step40_plot_performance.py --base-dir {args.run_dir} --split-seed {top_seed}"
                f"\n{'='*80}"
            )
        
        logger.info("\n✓ Processing complete")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
