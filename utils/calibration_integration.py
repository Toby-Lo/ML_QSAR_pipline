"""
Integration module: Calibration-aware AD analysis for Step22

This module provides functions to automatically detect and integrate Step20's
calibrated predictions into Step22's applicability domain analysis.

Usage in step22_applicability_domain.py:
    from utils.calibration_integration import (
        try_load_calibrated_predictions,
        flexible_column_resolution,
        compute_calibration_comparison_curve,
    )
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd


# ============================================================================
# PART A: Calibration Detection & Loading
# ============================================================================

def try_load_calibrated_predictions(
    run_dir: Path,
    split_seed: int,
    model_key: str,
    calibration_method: Optional[str] = None,
) -> Optional[Tuple[Any, Dict[str, Any]]]:
    """
    尝试从 Step20 的校准结果中加载校准后的模型。
    
    在以下位置查找校准模型：
        {run_dir}/split_seed_{split_seed}/calibration/{model_key}/method_{method}/
        
    检查顺序：
        1. 若指定了 calibration_method，则优先查找该方法
        2. 否则按 ["sigmoid", "isotonic"] 顺序查找
        
    Args:
        run_dir: 运行目录（例如 models_out/qsar_ml_20260412_162829）
        split_seed: 分割种子
        model_key: 模型名称（例如 "SVC", "RFC", "MLP"）
        calibration_method: 优先使用的校准方法 ("sigmoid" 或 "isotonic")
        
    Returns:
        成功时: (calibrated_model, metadata_dict)
            其中 metadata_dict 包含:
                - calibration_source: str ("step20/sigmoid" 或 "step20/isotonic")
                - calibration_method_used: str
                - brier_raw: float
                - brier_calibrated: float
                - brier_improvement: float
                
        失败时: None (所有方法都不存在或无法加载)
        
    Examples:
        >>> result = try_load_calibrated_predictions(
        ...     run_dir=Path("models_out/qsar_ml_20260412_162829"),
        ...     split_seed=12345,
        ...     model_key="SVC",
        ...     calibration_method="sigmoid"
        ... )
        >>> if result is not None:
        ...     calibrated_model, metadata = result
        ...     print(f"Calibration method: {metadata['calibration_method_used']}")
    """
    split_dir = run_dir / f"split_seed_{split_seed}"
    cal_dir = split_dir / "calibration" / model_key
    
    if not cal_dir.exists():
        logging.debug(f"No calibration directory found: {cal_dir}")
        return None
    
    # 构建要尝试的方法列表
    methods_to_try: List[str] = []
    if calibration_method:
        if calibration_method not in ["sigmoid", "isotonic"]:
            raise ValueError(f"Unknown calibration_method: {calibration_method}")
        methods_to_try.append(calibration_method)
    
    methods_to_try.extend(["sigmoid", "isotonic"])
    
    # 尝试每个方法
    for method in methods_to_try:
        method_dir = cal_dir / f"method_{method}"
        
        if not method_dir.exists():
            logging.debug(f"Method directory not found: {method_dir}")
            continue
        
        model_path = method_dir / "calibrated_model.joblib"
        metrics_path = method_dir / "calibration_metrics.json"
        
        if not model_path.exists():
            logging.debug(f"Model file not found: {model_path}")
            continue
        
        try:
            # 加载校准模型
            calibrated_model = joblib.load(model_path)
            
            # 加载元数据（Brier scores 等）
            metadata: Dict[str, Any] = {
                "calibration_source": f"step20/{method}",
                "calibration_method_used": method,
            }
            
            if metrics_path.exists():
                try:
                    metrics = json.loads(metrics_path.read_text())
                    metadata.update(metrics)
                except Exception as e:
                    logging.warning(f"Failed to load metrics from {metrics_path}: {e}")
            
            logging.info(
                f"✓ Loaded calibrated model from step20/{method} "
                f"(split_seed={split_seed}, model={model_key})"
            )
            
            return (calibrated_model, metadata)
        
        except Exception as e:
            logging.warning(
                f"Failed to load calibrated model from {method_dir}: {e}. "
                f"Trying next method..."
            )
            continue
    
    logging.debug(
        f"✗ No calibrated model found for split_seed={split_seed}, model={model_key}"
    )
    return None


def apply_calibrated_model_to_external(
    calibrated_model: Any,
    X_external: np.ndarray,
    external_ids: np.ndarray,
    external_smiles: np.ndarray,
    original_predictions_csv: Path,
    model_key: str,
    id_column: str = "id",
    smiles_column: str = "smiles",
    model_column: str = "model",
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    使用校准模型对外部测试集生成校准后的概率。
    
    Step 1: 使用 calibrated_model.predict_proba() 生成新的概率
    Step 2: 与原始预测 CSV 进行 ID/SMILES 双键验证
    Step 3: 返回校准概率数组和对齐后的预测表
    
    Args:
        calibrated_model: 从 try_load_calibrated_predictions() 获得的模型
        X_external: 外部测试集特征矩阵 (shape: N×D)
        external_ids: 外部样本的 ID (shape: N,)
        external_smiles: 外部样本的 SMILES (shape: N,)
        original_predictions_csv: 原始预测 CSV 文件路径
        model_key: 筛选 CSV 中的模型列时使用
        id_column: ID 列名
        smiles_column: SMILES 列名
        model_column: 模型列名
        
    Returns:
        (calibrated_probs, merged_df)
            - calibrated_probs: 校准后的概率数组 (shape: N,)
            - merged_df: 对齐后的 DataFrame（包含原始和校准概率）
            
    Raises:
        ValueError: 若 ID/SMILES 对齐失败
    """
    # Step 1: 生成校准概率
    try:
        calibrated_probs = calibrated_model.predict_proba(X_external)[:, 1]
        calibrated_probs = calibrated_probs.astype(np.float64)
    except Exception as e:
        raise ValueError(
            f"Failed to generate predictions from calibrated model: {e}. "
            f"Check that X_external has correct shape and features."
        ) from e
    
    # Step 2: 读取原始预测
    if not original_predictions_csv.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {original_predictions_csv}")
    
    preds_df = pd.read_csv(original_predictions_csv)
    
    # 筛选指定模型的行
    model_mask = preds_df[model_column].astype(str).str.upper() == str(model_key).upper()
    preds_df = preds_df[model_mask].copy()
    
    if preds_df.empty:
        raise ValueError(
            f"No predictions found for model '{model_key}' in {original_predictions_csv}"
        )
    
    # Step 3: 创建校准概率的临时表
    temp_df = pd.DataFrame({
        id_column: external_ids.astype(str),
        smiles_column: external_smiles.astype(str),
        "y_prob_calibrated": calibrated_probs,
    })
    
    # Step 4: 按 ID + SMILES 双键合并
    try:
        merged = temp_df.merge(
            preds_df[[id_column, smiles_column, "y_prob"]],
            on=[id_column, smiles_column],
            how="left",
            validate="one_to_one"
        )
    except pd.errors.MergeError as e:
        # Fallback: try ID-only merge if smiles diverge
        logging.warning(f"ID+SMILES merge failed: {e}. Trying ID-only merge...")
        merged = temp_df[[id_column]].merge(
            preds_df[[id_column, "y_prob"]],
            on=[id_column],
            how="left"
        )
        if smiles_column in preds_df.columns:
            merged = merged.merge(
                temp_df[[id_column, smiles_column, "y_prob_calibrated"]],
                on=[id_column],
                how="left"
            )
    
    # Step 5: 验证对齐完整性
    if merged.isna().any().any():
        missing_count = merged.isna().sum().sum()
        raise ValueError(
            f"Failed to align calibrated predictions with original predictions. "
            f"{missing_count} NaN values after merge (out of {len(merged)} rows). "
            f"Check whether external IDs/SMILES match predictions file ordering."
        )
    
    logging.info(
        f"✓ Successfully generated {len(calibrated_probs)} calibrated predictions "
        f"and aligned with original predictions"
    )
    
    return calibrated_probs, merged


# ============================================================================
# PART B: Flexible Column Resolution
# ============================================================================

def flexible_column_resolution(
    df: pd.DataFrame,
    priority_names: List[str],
    required: bool = True,
    allow_freeform: bool = False,
    friendly_name: str = "column",
) -> Tuple[str, np.ndarray]:
    """
    灵活地查找和返回列数据，按优先级顺序检查候选列名。
    
    解决问题：
        - 若 Step20 修改了列名（如 y_prob → y_prob_calibrated），
          确保 Step22 能自动找到新列名
        - 优先级逻辑：优先校准版本，然后原始版本
        
    Args:
        df: 输入 DataFrame
        priority_names: 列名候选列表，按优先级排序
            例如: ["y_prob_calibrated", "y_prob_cal", "y_prob"]
        required: 若 True，未找到时抛异常；若 False，返回 (None, None)
        allow_freeform: 若 True 且找不到时，允许用户输入列名
        friendly_name: 用于错误消息和日志的友好名称
        
    Returns:
        (實際使用的列名, 列数据数组)
        - 列数据已转为 np.ndarray, dtype=float64
        
    Raises:
        ValueError: 若 required=True 且找不到列
        
    Examples:
        >>> df = pd.DataFrame({"y_prob_calibrated": [0.1, 0.9], "y_prob": [0.2, 0.8]})
        >>> col_name, col_data = flexible_column_resolution(
        ...     df,
        ...     priority_names=["y_prob_calibrated", "y_prob"],
        ...     friendly_name="predicted probability"
        ... )
        >>> print(col_name)
        'y_prob_calibrated'
    """
    for col in priority_names:
        if col in df.columns:
            try:
                col_data = df[col].astype(np.float64).to_numpy()
                logging.info(
                    f"✓ Using column '{col}' for {friendly_name} "
                    f"(matched priority #{priority_names.index(col) + 1})"
                )
                return col, col_data
            except Exception as e:
                logging.warning(
                    f"Found column '{col}' for {friendly_name}, "
                    f"but conversion failed: {e}. Trying next..."
                )
                continue
    
    # 所有优先级都失败
    if required:
        msg = (
            f"Could not find {friendly_name} column in DataFrame.\n"
            f"Searched for (in priority order): {priority_names}\n"
            f"Available columns: {list(df.columns)}"
        )
        raise ValueError(msg)
    else:
        logging.warning(f"Could not find {friendly_name} column, returning None")
        return None, None


# ============================================================================
# PART C: Calibration Diagnostics & Curve Computation
# ============================================================================

def compute_calibration_comparison_curve(
    ad_score: np.ndarray,
    error_raw: np.ndarray,
    error_calibrated: Optional[np.ndarray] = None,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    对比 AD_Score 对原始错误与校准错误的预测能力。
    
    This function addresses Task 4 enhancement: quantifying how AD improves
    when using calibrated vs uncalibrated probabilities.
    
    Args:
        ad_score: AD 得分 (shape: N,)
        error_raw: 原始预测的错误度量（如 log-loss）(shape: N,)
        error_calibrated: 校准后预测的错误度量 (shape: N,)，可选
        n_bins: 将 AD_Score [0,1] 分成多少个 bin
        
    Returns:
        DataFrame with columns:
            - ad_min, ad_max, ad_mean: AD 分数的 bin 范围和平均值
            - error_raw_mean, error_raw_std: 原始错误的统计
            - error_calibrated_mean, error_calibrated_std: 校准错误的统计（若提供）
            - error_improvement: 错误改进幅度 (raw - calibrated)
            - count: 该 bin 中的样本数
            
    Example:
        >>> curve_df = compute_calibration_comparison_curve(
        ...     ad_score=np.array([0.2, 0.5, 0.8]),
        ...     error_raw=np.array([0.5, 0.3, 0.1]),
        ...     error_calibrated=np.array([0.45, 0.25, 0.08]),
        ...     n_bins=3
        ... )
        >>> print(curve_df)
           ad_min  ad_max  ad_mean  error_raw_mean  error_calibrated_mean  error_improvement  count
        0    0.0  0.333      ...              ...                      ...                ...      1
        1    0.333  0.667  ...              ...                      ...                ...      1
        2    0.667  1.0    ...              ...                      ...                ...      1
    """
    ad_score = np.asarray(ad_score, dtype=np.float64)
    error_raw = np.asarray(error_raw, dtype=np.float64)
    
    if error_calibrated is not None:
        error_calibrated = np.asarray(error_calibrated, dtype=np.float64)
    
    results: List[Dict[str, Any]] = []
    
    for i in range(n_bins):
        ad_min = i / n_bins
        ad_max = (i + 1) / n_bins
        
        mask = (ad_score >= ad_min) & (ad_score < ad_max)
        
        if not np.any(mask):
            continue  # Skip empty bins
        
        row: Dict[str, Any] = {
            "ad_min": float(ad_min),
            "ad_max": float(ad_max),
            "ad_mean": float(np.mean(ad_score[mask])),
            "error_raw_mean": float(np.mean(error_raw[mask])),
            "error_raw_std": float(np.std(error_raw[mask])),
            "count": int(np.sum(mask)),
        }
        
        # 如果提供了校准错误信息
        if error_calibrated is not None:
            row["error_calibrated_mean"] = float(np.mean(error_calibrated[mask]))
            row["error_calibrated_std"] = float(np.std(error_calibrated[mask]))
            row["error_improvement"] = float(
                row["error_raw_mean"] - row["error_calibrated_mean"]
            )
        
        results.append(row)
    
    return pd.DataFrame(results)


def compute_calibration_diagnostics(
    y_prob_raw: np.ndarray,
    y_prob_calibrated: np.ndarray,
    y_true: np.ndarray,
    eps: float = 1e-9,
) -> Dict[str, Any]:
    """
    计算校准前后的诊断统计。
    
    Args:
        y_prob_raw: 原始模型概率 (shape: N,)
        y_prob_calibrated: 校准后的概率 (shape: N,)
        y_true: 真实标签 (shape: N,)
        eps: 数值稳定性参数
        
    Returns:
        诊断字典:
            - log_loss_raw: 原始log损失
            - log_loss_calibrated: 校准log损失
            - brier_raw: 原始 Brier 得分
            - brier_calibrated: 校准 Brier 得分
            - ece_raw: 期望校准误差（原始）
            - ece_calibrated: 期望校准误差（校准）
    """
    from sklearn.metrics import log_loss, brier_score_loss
    
    diagnostics = {
        "log_loss_raw": float(log_loss(y_true, y_prob_raw)),
        "log_loss_calibrated": float(log_loss(y_true, y_prob_calibrated)),
        "brier_raw": float(brier_score_loss(y_true, y_prob_raw)),
        "brier_calibrated": float(brier_score_loss(y_true, y_prob_calibrated)),
    }
    
    # Expected Calibration Error
    from sklearn.calibration import calibration_curve
    
    frac_raw, mean_raw = calibration_curve(y_true, y_prob_raw, n_bins=10)
    frac_cal, mean_cal = calibration_curve(y_true, y_prob_calibrated, n_bins=10)
    
    ece_raw = float(np.mean(np.abs(mean_raw - frac_raw)))
    ece_cal = float(np.mean(np.abs(mean_cal - frac_cal)))
    
    diagnostics["ece_raw"] = ece_raw
    diagnostics["ece_calibrated"] = ece_cal
    
    return diagnostics


# ============================================================================
# PART D: Shrinkage Strategy Selection
# ============================================================================

def apply_ad_shrinkage(
    y_prob: np.ndarray,
    ad_score: np.ndarray,
    method: str = "probability_space",
) -> np.ndarray:
    """
    在选定的概率空间中应用 AD 收缩。
    
    处理 Task 3 中提出的问题：
        在概率空间 vs logit 空间应用 AD 收缩的折衷方案
        
    Args:
        y_prob: 模型输出的概率 (shape: N,)
        ad_score: 应用域得分 (shape: N,)，范围 [0, 1]
        method: 收缩方法
            - "probability_space": 直接乘积 p * AD_Score
              保留校准属性但可能过度收缩
            - "logit_space": 在 logit 空间内应用
              需要注意校准属性的破坏
            - "conservative": 最小化校准破坏
              混合方法：仅当 AD_Score < 0.5 时才应用
            - "none": 不应用收缩
              
    Returns:
        应用收缩后的得分 (shape: N,)
        
    Notes:
        - 若已应用 step20 校准，推荐使用 "probability_space"
        - 若为原始概率，可考虑 "logit_space" 获得更好的概率校准
    """
    y_prob = np.asarray(y_prob, dtype=np.float64)
    ad_score = np.asarray(ad_score, dtype=np.float64)
    
    y_prob = np.clip(y_prob, 0.0, 1.0)
    ad_score = np.clip(ad_score, 0.0, 1.0)
    
    if method == "probability_space":
        # 直接乘积，保留校准属性
        return y_prob * ad_score
    
    elif method == "logit_space":
        # Logit 空间（可能破坏校准）
        eps = 1e-9
        logit = np.log(np.clip(y_prob, eps, 1.0 - eps) / 
                       np.clip(1.0 - y_prob, eps, 1.0 - eps))
        logit_scaled = logit * ad_score
        logit_scaled = np.clip(logit_scaled, -500, 500)  # Numerical stability
        result = 1.0 / (1.0 + np.exp(-logit_scaled))
        return result
    
    elif method == "conservative":
        # 混合：仅当 AD_Score < 0.5 时才强收缩
        result = np.where(
            ad_score < 0.5,
            y_prob * ad_score,      # Strong shrinkage in low-confidence domain
            y_prob * (0.5 + 0.5 * ad_score)  # Gentle shrinkage in high-confidence
        )
        return result
    
    elif method == "none":
        # 直接使用 AD_Score 作为最终得分
        return ad_score
    
    else:
        raise ValueError(
            f"Unknown shrinkage method: {method}. "
            f"Choose from: ['probability_space', 'logit_space', 'conservative', 'none']"
        )


# ============================================================================
# Main Integration: wrap everything for step22
# ============================================================================

def integrate_calibration_to_ad_config(
    config_dict: Dict[str, Any],
    run_dir: Path,
    split_seed: int,
) -> Dict[str, Any]:
    """
    从 step20 的输出自动填充 step22 的配置。
    
    使用场景：
        1. 自动检测是否存在校准版本
        2. 推荐合适的 logit_shrinkage_method
        3. 提示用户是否应使用校准概率
        
    Args:
        config_dict: Step22 的配置字典（来自 USER_CONFIG）
        run_dir: 运行目录
        split_seed: 分割种子
        
    Returns:
        增强后的配置字典，包含以下新字段：
            - calibration_available: bool
            - calibration_metadata: dict 或 None
            - recommended_shrinkage_method: str
    """
    result = config_dict.copy()
    
    model_key = result.get("model_key", "SVC")
    
    # 检测校准
    calib_result = try_load_calibrated_predictions(
        run_dir=run_dir,
        split_seed=split_seed,
        model_key=model_key,
        calibration_method=result.get("calibration_method")
    )
    
    if calib_result is not None:
        _, metadata = calib_result
        result["calibration_available"] = True
        result["calibration_metadata"] = metadata
        result["recommended_shrinkage_method"] = "probability_space"  # 安全选项
        
        logging.info(
            f"✓ Calibration detected from step20 for {model_key}. "
            f"Recommended shrinkage method: probability_space (to preserve calibration)"
        )
    else:
        result["calibration_available"] = False
        result["calibration_metadata"] = None
        result["recommended_shrinkage_method"] = "conservative"  # 中立选项
        
        logging.info(
            f"No calibration found for {model_key}. "
            f"Using uncalibrated probabilities (default behavior)."
        )
    
    return result


if __name__ == "__main__":
    # 简单测试
    logging.basicConfig(level=logging.INFO)
    
    # 例子相关的测试代码
    print("[OK] Integration module loaded successfully")
