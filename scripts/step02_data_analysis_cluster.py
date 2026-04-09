# %% 
# 1. Import Libraries and Configuration
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.decomposition import PCA  # 新增 PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

# Set global font to Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
sns.set_theme(style="white", font='Times New Roman') # 改为 white 风格更学术

# %% 
# 2. Load Data and Process Fingerprints
file_path = '../Data/NSD2/NSD2_final_ic50_with_fingerprints.csv'  ###### Adjust this path as needed
df = pd.read_csv(file_path)
print(f"Dataset loaded with shape: {df.shape}")

# 生成或提取指纹
fp_cols = [c for c in df.columns if c.startswith('morgan_')]
if len(fp_cols) >= 2048:
    X = df[fp_cols].values
else:
    def get_fp(smile):
        mol = Chem.MolFromSmiles(smile)
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
    X = np.array([get_fp(s) for s in df['smiles']])

X = X.astype(int)
print(f"Data matrix shape: {X.shape}")

# %% 
# 3. Two-Step Dimension Reduction: PCA then t-SNE
print("Step 1: Running PCA to reduce noise (Keeping 50 components)...")
# 先用 PCA 降到 50 维，这是处理高维指纹的 standard protocol
pca = PCA(n_components=50, random_state=42)
X_pca = pca.fit_transform(X)

print("Step 2: Running t-SNE for local structure visualization...")
# 基于 PCA 的结果跑 t-SNE
tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
X_tsne = tsne.fit_transform(X_pca)

df['tsne_1'] = X_tsne[:, 0]
df['tsne_2'] = X_tsne[:, 1]
print("Dimension reduction completed.")

# %%
# 诊断 PCA 的数据
print(f"Max TSNE 1 value: {df['tsne_1'].max()}")
print(f"Min TSNE 1 value: {df['tsne_1'].min()}")

# %% 
# 4. Clustering Evaluation 
print("Evaluating optimal cluster number (k)...")
k_range = range(2, 11)
inertia = []
silhouette_avg = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_tsne)
    inertia.append(kmeans.inertia_)
    silhouette_avg.append(silhouette_score(X_tsne, labels))

# --- 绘图部分：双轴评估图 ---
fig, ax1 = plt.subplots(figsize=(8, 5), dpi=150)

ax1.plot(k_range, inertia, marker='o', color='#1f77b4', lw=2, label='Inertia (Elbow)')
ax1.set_xlabel('Number of Clusters (k)', fontweight='bold')
ax1.set_ylabel('Inertia', color='#1f77b4', fontweight='bold')
ax1.tick_params(axis='y', labelcolor='#1f77b4', direction='in')
ax1.tick_params(axis='x', direction='in')

ax2 = ax1.twinx()
ax2.plot(k_range, silhouette_avg, marker='s', color='#d62728', lw=2, ls='--', label='Silhouette')
ax2.set_ylabel('Silhouette Score', color='#d62728', fontweight='bold')
ax2.tick_params(axis='y', labelcolor='#d62728', direction='in')

# 强化 L 型边框
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

plt.title('Optimization of Cluster Number (k)', fontweight='bold', pad=15)
plt.tight_layout()
plt.show()
#plt.savefig("../Data/NSD2/Cluster_Optimization_k.png", dpi=300, bbox_inches='tight')

# %%
# 5. Final K-Means Clustering
manual_k = 6 
kmeans_final = KMeans(n_clusters=manual_k, random_state=42, n_init=10)
df['cluster'] = kmeans_final.fit_predict(X_tsne)


# %% 
# 6. Final Visualization: Dual-Plot (Clusters with Centroids vs. Bioactivity)
import matplotlib.lines as mlines

# --- 1. 数据预处理与强转 ---
df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
df['cluster'] = pd.to_numeric(df['cluster'], errors='coerce').fillna(0).astype(int)

# 获取质心坐标 (基于 KMeans 结果)
centroids = kmeans_final.cluster_centers_

output_base = f"../Data/NSD2/NSD2_Dual_Analysis_k{manual_k}"

# --- 2. 创建双子图 (学术比例 18:8) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), dpi=300)

# 颜色配置
cluster_palette = sns.color_palette("husl", manual_k) 
activity_palette = {1: '#28559A', 0: '#B73131'} # 深蓝与深红

# --- A. 左子图：Scaffold Clusters + Centroids ---
for i in range(manual_k):
    subset = df[df['cluster'] == i]
    if not subset.empty:
        ax1.scatter(
            subset['tsne_1'], subset['tsne_2'], 
            color=cluster_palette[i],
            s=40, alpha=0.5, edgecolors='white', linewidths=0.2,
            label=f'Cluster {i}'
        )

# 绘制质心 - 采用双层叠加法增强视觉对比
'''
ax1.scatter(
    centroids[:, 0], centroids[:, 1], 
    s=250, c='none', edgecolors='black', marker='o', linewidths=2, alpha=0.8, zorder=10
)
'''
ax1.scatter(
    centroids[:, 0], centroids[:, 1], 
    s=120, c='black', marker='x', linewidths=2.5, label='Centroids', zorder=11
)
ax1.set_title('(A) Chemical Scaffold Clusters', fontsize=18, fontweight='bold', loc='left', pad=15)
ax1.set_xlabel('t-SNE Dimension 1', fontsize=14, fontweight='bold')
ax1.set_ylabel('t-SNE Dimension 2', fontsize=14, fontweight='bold')
ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, title="Groups")

# --- B. 右子图：Bioactivity Distribution ---
for val in [0, 1]:
    subset = df[df['label'] == val]
    if not subset.empty:
        ax2.scatter(
            subset['tsne_1'], subset['tsne_2'], 
            color=activity_palette[val],
            s=45, alpha=0.6, edgecolors='white', linewidths=0.3,
            zorder=3 if val == 1 else 2, # 活性点置顶
            label='Active (1)' if val == 1 else 'Inactive (0)'
        )

ax2.set_title('(B) Bioactivity Distribution', fontsize=18, fontweight='bold', loc='left', pad=15)
ax2.set_xlabel('t-SNE Dimension 1', fontsize=14, fontweight='bold')
ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, title="Activity")

# --- 3. 统一学术格式化：强化 L 型轴线 ---
for ax in [ax1, ax2]:
    # 移除冗余边框
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # 强化 L 型边框线宽
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    # 刻度线向内
    ax.tick_params(direction='in', length=6, width=1.5, labelsize=12)
    ax.grid(False)

plt.tight_layout()

# --- 4. 多格式导出 ---
# 导出 PNG 用于常规预览 (600 DPI)
plt.savefig(f"{output_base}.png", dpi=600, bbox_inches='tight')
# 导出 SVG 用于论文排版 (矢量无损)
plt.savefig(f"{output_base}.svg", format='svg', bbox_inches='tight')

plt.show()

print(f"Success! \nPNG: {output_base}.png \nSVG: {output_base}.svg")
# %%
