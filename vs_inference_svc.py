#!/usr/bin/env python3
"""
虚拟筛选推断脚本 - 使用SVC模型
==============================

使用step10训练的最优SVC模型进行新化合物的虚拟筛选

使用方法:
    python vs_inference_svc.py --smiles "O=C(O)c1ccccc1" "CCO" ...
    或从CSV读取:
    python vs_inference_svc.py --input candidates.csv --smiles_col smiles
"""

import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors


# 配置 - 根据您的运行修改
CONFIG = {
    'run_dir': 'models_out/qsar_ml_20260410_124055',
    'split_seed': 26,  # 推荐种子
    'fp_radius': 2,
    'fp_nbits': 2048,
    'descriptor_names': [
        'MolWt', 'MolLogP', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds',
        'TPSA', 'HeavyAtomCount', 'NumValenceElectrons', 'NumAliphaticRings',
        'NumAromaticRings', 'FractionCSP3', 'RingCount', 'LabuteASA',
        'VSA_EState1', 'VSA_EState2', 'SlogP_VSA1', 'SlogP_VSA2',
    ]
}


class SVCVirtualScreening:
    """虚拟筛选推断器"""

    def __init__(self, run_dir: str, split_seed: int = 26):
        """加载模型和特征配置"""
        self.run_dir = Path(run_dir)
        self.seed_dir = self.run_dir / f"split_seed_{split_seed}"

        # 加载模型
        model_path = self.seed_dir / "models" / "SVC" / f"seed_{split_seed}" / "model.joblib"
        self.model = joblib.load(model_path)
        print(f"✓ 加载SVC模型: {model_path}")

        # 加载缩放器
        scaler_path = self.seed_dir / "models" / "SVC" / f"seed_{split_seed}" / "scaler.joblib"
        self.scaler = joblib.load(scaler_path)
        print(f"✓ 加载描述符缩放器: {scaler_path}")

        # 加载指纹掩码
        mask_path = self.seed_dir / "feature_processors" / "fp_mask.npy"
        self.fp_mask = np.load(mask_path)
        print(f"✓ 加载指纹掩码 ({np.sum(self.fp_mask)} / 2048 bits)")

        # 加载阈值
        threshold_path = self.seed_dir / "results" / "threshold_selection_summary.csv"
        if threshold_path.exists():
            threshold_df = pd.read_csv(threshold_path)
            svc_row = threshold_df[threshold_df['model'] == 'SVC'].iloc[0]
            self.threshold = svc_row.get('selected_threshold', 0.5)
            print(f"✓ 加载锁定阈值: {self.threshold:.4f}")
        else:
            self.threshold = 0.5
            print(f"⚠️  使用默认阈值: {self.threshold:.4f}")

    def compute_fingerprints(self, smiles_list: List[str]) -> Optional[np.ndarray]:
        """计算Morgan指纹"""
        fingerprints = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                print(f"⚠️  无效的SMILES: {smi}")
                return None

            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, CONFIG['fp_radius'], nBits=CONFIG['fp_nbits']
            )
            fingerprints.append(np.array(fp, dtype=np.float32))

        return np.array(fingerprints, dtype=np.float32)

    def compute_descriptors(self, smiles_list: List[str]) -> Optional[np.ndarray]:
        """计算RDKit描述符"""
        descriptors = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return None

            desc = []
            for name in CONFIG['descriptor_names']:
                try:
                    func = getattr(Descriptors, name)
                    value = float(func(mol)) if mol is not None else float('nan')
                except Exception:
                    value = float('nan')
                desc.append(value)

            descriptors.append(desc)

        return np.array(descriptors, dtype=np.float32)

    def prepare_features(self, smiles_list: List[str]) -> Optional[np.ndarray]:
        """准备特征 (指纹 + 描述符)"""
        # 计算指纹
        fp_matrix = self.compute_fingerprints(smiles_list)
        if fp_matrix is None:
            return None

        # 应用掩码过滤
        fp_filtered = fp_matrix[:, self.fp_mask].astype(np.float32)

        # 计算描述符
        desc_matrix = self.compute_descriptors(smiles_list)
        if desc_matrix is None:
            return None

        # 缩放描述符
        desc_scaled = self.scaler.transform(desc_matrix).astype(np.float32)

        # 拼接
        X = np.concatenate([fp_filtered, desc_scaled], axis=1).astype(np.float32)
        return X

    def predict(self, smiles_list: List[str]) -> pd.DataFrame:
        """进行虚拟筛选预测"""
        # 准备特征
        X = self.prepare_features(smiles_list)
        if X is None:
            raise ValueError("特征准备失败")

        # 预测
        y_prob = self.model.predict_proba(X)[:, 1]
        y_pred = (y_prob >= self.threshold).astype(int)

        # 组织结果
        results = pd.DataFrame({
            'smiles': smiles_list,
            'probability': y_prob,
            'prediction': y_pred,
            'active': y_pred == 1,
        })

        # 按概率降序排列
        results = results.sort_values('probability', ascending=False).reset_index(drop=True)

        return results


def main():
    parser = argparse.ArgumentParser(description='虚拟筛选推断')
    parser.add_argument('--smiles', nargs='+', help='SMILES列表')
    parser.add_argument('--input', type=str, help='输入CSV文件')
    parser.add_argument('--smiles-col', type=str, default='smiles', help='SMILES列名')
    parser.add_argument('--output', type=str, default='vs_results.csv', help='输出文件')
    parser.add_argument('--run-dir', type=str, default=CONFIG['run_dir'], help='运行目录')
    parser.add_argument('--seed', type=int, default=CONFIG['split_seed'], help='分割种子')
    parser.add_argument('--top-n', type=int, help='仅显示top N个')

    args = parser.parse_args()

    # 加载模型
    vscreen = SVCVirtualScreening(args.run_dir, args.seed)

    # 读取SMILES
    if args.input:
        df = pd.read_csv(args.input)
        smiles_list = df[args.smiles_col].tolist()
        print(f"✓ 从 {args.input} 读取 {len(smiles_list)} 个化合物")
    elif args.smiles:
        smiles_list = args.smiles
        print(f"✓ 使用 {len(smiles_list)} 个提供的SMILES")
    else:
        parser.print_help()
        return

    # 进行预测
    print("\n进行虚拟筛选...)
    results = vscreen.predict(smiles_list)

    # 显示结果
    print("\n" + "=" * 100)
    print("虚拟筛选结果")
    print("=" * 100)

    if args.top_n:
        display = results.head(args.top_n)
    else:
        display = results

    print(display.to_string(index=False))

    # 统计
    active_count = (results['prediction'] == 1).sum()
    print(f"\n总计: {len(results)} 个, 预测活性: {active_count} 个 ({100*active_count/len(results):.1f}%)")

    # 保存结果
    results.to_csv(args.output, index=False)
    print(f"\n✓ 结果已保存到: {args.output}")

    # 显示top候选
    print("\n" + "=" * 100)
    print("Top 候选 (预测为活性，按概率排序)")
    print("=" * 100)
    top_actives = results[results['prediction'] == 1].head(10)
    print(top_actives.to_string(index=False))


if __name__ == '__main__':
    main()
