import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
from tqdm import tqdm

def generate_scaffold_smiles(smiles):
    """生成分子的 Murcko Scaffold SMILES 字符串"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    # 关键修改：获取 Scaffold 的 Mol 后，必须转回 SMILES 字符串
    scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold_mol)

def convert_split_datasets(csv_path, output_dir, split_seed=12345, train_size=0.8):
    # 1. 读取原始 CSV
    df = pd.read_csv(csv_path)
    print(f"读取完成，原始形状: {df.shape}")

    # 2. 复现 Scaffold Split 逻辑
    print(f"正在计算分子的 Scaffolds (Split Seed: {split_seed})...")
    scaffolds = defaultdict(list)
    
    # 遍历所有分子，根据 Scaffold SMILES 进行归类
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        scaffold_smiles = generate_scaffold_smiles(row['smiles'])
        scaffolds[scaffold_smiles].append(idx)
    
    # 排序：这次是对 SMILES 字符串排序，不会报错
    scaffold_list = sorted(scaffolds.keys())
    
    # 使用训练时确定的随机种子进行打乱
    rng = np.random.RandomState(split_seed)
    rng.shuffle(scaffold_list)
    
    train_indices = []
    n_train_target = int(len(df) * train_size)
    
    # 严格遵循训练脚本的分配逻辑
    for scaffold in scaffold_list:
        if len(train_indices) + len(scaffolds[scaffold]) <= n_train_target:
            train_indices.extend(scaffolds[scaffold])
        else:
            # 一旦达到比例，剩余骨架全部分配给测试集
            break
    
    test_indices = list(set(df.index) - set(train_indices))
    
    df_train = df.iloc[train_indices].copy()
    df_test = df.iloc[test_indices].copy()

    # 3. 规范化处理以适配推理脚本
    def finalize_df(temp_df):
        # 将原始索引保存为 zinc_id，确保是 int64 类型
        temp_df['zinc_id'] = temp_df.index.astype('int64')
        # 填充数值列缺失值
        num_cols = temp_df.select_dtypes(include=[np.number]).columns
        temp_df[num_cols] = temp_df[num_cols].fillna(0.0)
        # 确保数据格式
        temp_df['smiles'] = temp_df['smiles'].astype(str)
        return temp_df

    df_train = finalize_df(df_train)
    df_test = finalize_df(df_test)

    # 4. 保存为 Parquet
    train_path = f"{output_dir}/nsd2_dev_set_seed{split_seed}.parquet"
    test_path = f"{output_dir}/nsd2_test_set_seed{split_seed}.parquet"
    
    pq.write_table(pa.Table.from_pandas(df_train, preserve_index=False), train_path, compression="zstd")
    pq.write_table(pa.Table.from_pandas(df_test, preserve_index=False), test_path, compression="zstd")
    
    print(f"\n✅ 划分完成 (Split Seed: {split_seed})")
    print(f"--- 训练集 (Development): {len(df_train)} 样本 -> {train_path}")
    print(f"--- 测试集 (External): {len(df_test)} 样本 -> {test_path}")
    return train_path, test_path

if __name__ == "__main__":
    convert_split_datasets(
        csv_path="./data/NSD2/nsd2_final_dataset_feature_fingerprint.csv",
        output_dir="./data/NSD2",
        split_seed=12345
    )