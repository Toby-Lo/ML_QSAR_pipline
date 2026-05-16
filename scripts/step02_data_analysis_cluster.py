# %%
# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from pathlib import Path

# %%
# 2. Style & Output Config
sns.set_theme(style="white")

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Cambria', 'Times New Roman', 'Times', 'DejaVu Serif'],
    'svg.fonttype': 'none',
})

# 输出目录
output_dir = Path("../data/NSD2/cluster")
output_dir.mkdir(parents=True, exist_ok=True)

# %%
# 3. Load Data
file_path = '../data/NSD2/nsd2_final_dataset.csv'
df = pd.read_csv(file_path)

df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)

print(f"Dataset loaded: {df.shape}")

# %%
# 4. Morgan Fingerprints
fpgen = GetMorganGenerator(radius=2, fpSize=2048)

def get_fp(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return np.zeros(2048, dtype=int)
    
    fp = fpgen.GetFingerprint(mol)
    arr = np.zeros((2048,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

print("Generating fingerprints...")
X = np.vstack([get_fp(s) for s in df['smiles']])
print(f"Fingerprint matrix: {X.shape}")

# %%
# 5. Standardization + PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=50, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f"Explained variance (50 PCs): {pca.explained_variance_ratio_.sum():.3f}")

# %%
# 6. PCA Variance Curve
plt.figure(figsize=(6,4))
plt.plot(np.cumsum(pca.explained_variance_ratio_)*100, lw=2)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance (%)')
plt.title('PCA Explained Variance')

plt.tight_layout()
plt.savefig(output_dir / "Figure_S1_PCA_variance.svg", bbox_inches='tight')
plt.show()

# %%
# 7. PCA Scatter
df['PC1'] = X_pca[:, 0]
df['PC2'] = X_pca[:, 1]

palette = {1: '#28559A', 0: '#B73131'}

fig, ax = plt.subplots(figsize=(6,5))

for val in [0,1]:
    subset = df[df['label']==val]
    ax.scatter(subset['PC1'], subset['PC2'],
               color=palette[val],
               s=20, alpha=0.6,
               label='Active' if val==1 else 'Inactive')

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title('PCA Projection', fontweight='bold')

ax.legend(frameon=True)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(direction='out', length=5, width=1)

plt.tight_layout()
plt.savefig(output_dir / "Figure_S1_PCA_scatter.svg", bbox_inches='tight')
plt.show()

# %%
# 8. t-SNE
tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42,
    init='pca',
    learning_rate='auto'
)

X_tsne = tsne.fit_transform(X_pca)

df['tsne_1'] = X_tsne[:, 0]
df['tsne_2'] = X_tsne[:, 1]

# %%
# 9. Clustering Evaluation（在 PCA 空间）
k_range = range(2, 11)
inertia = []
silhouette_avg = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)
    
    inertia.append(kmeans.inertia_)
    silhouette_avg.append(silhouette_score(X_pca, labels))

# Plot
fig, ax1 = plt.subplots(figsize=(7,5))

ax1.plot(k_range, inertia, 'o-', lw=2)
ax1.set_xlabel('k')
ax1.set_ylabel('Inertia')

ax2 = ax1.twinx()
ax2.plot(k_range, silhouette_avg, 's--', lw=2)
ax2.set_ylabel('Silhouette Score')

ax1.set_title('Cluster Optimization')

plt.tight_layout()
plt.savefig(output_dir / "Figure_S2_cluster_optimization.svg", bbox_inches='tight')
plt.show()

# %%
# 10. Final Clustering
manual_k = 8 # adjust based on elbow/silhouette

kmeans_final = KMeans(n_clusters=manual_k, random_state=42, n_init=10)
df['cluster'] = kmeans_final.fit_predict(X_pca)
df['cluster_id'] = df['cluster'] + 1

# %%
# 11. Final Visualization（t-SNE）
import matplotlib.ticker as ticker

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Cambria','Times New Roman'],
    'axes.titlesize': 15,
    'axes.labelsize': 13,
    'savefig.dpi': 600,
})

_x_min, _x_max = df['tsne_1'].min(), df['tsne_1'].max()
_y_min, _y_max = df['tsne_2'].min(), df['tsne_2'].max()
_x_range = _x_max - _x_min
_y_range = _y_max - _y_min

_pad = 0.01
_xlim = (_x_min - _pad * _x_range, _x_max + _pad * _x_range)
_ylim = (_y_min - _pad * _y_range, _y_max + _pad * _y_range)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), dpi=600, 
                               sharey=True, 
                               gridspec_kw={'wspace': 0.02})

# --- A. Clusters ---
cluster_colors = sns.color_palette("Set2", manual_k) 

for i in range(manual_k):
    subset = df[df['cluster'] == i]
    ax1.scatter(
        subset['tsne_1'], subset['tsne_2'],
        color=cluster_colors[i],
        s=65, alpha=1,
        edgecolors='white', linewidths=0.2,
        label=f'C{i+1}'
    )

ax1.set_title('(A)  Structural Clusters', loc='left', fontweight='bold', pad=15)
ax1.set_xlabel('t-SNE dimension 1')
ax1.set_ylabel('t-SNE dimension 2')

# legend
ax1.legend(frameon=True, ncol=1, loc='upper left', markerscale=1, 
           facecolor='white', edgecolor='black', framealpha=1,
           handletextpad=0.1, labelspacing = 0.3, borderpad=0.4
           )

# --- B. Activity ---
activity_colors = {1: '#28559A', 0: '#BDBDBD'} 

for val in [0, 1]:
    subset = df[df['label'] == val]
    ax2.scatter(
        subset['tsne_1'], subset['tsne_2'],
        color=activity_colors[val],
        #s=20 if val == 1 else 12,
        s=50,
        #alpha=0.8 if val == 1 else 0.4,
        alpha=1,
        edgecolors='white',linewidths=0.2,
        label='Active' if val == 1 else 'Inactive',
        zorder=2 if val == 1 else 1
    )

ax2.set_title('(B)  Bioactivity Distribution', loc='left', fontweight='bold', pad=15)
ax2.set_xlabel('t-SNE dimension 1')
ax2.set_yticks([]) # 保持整洁，隐藏右图 Y 轴刻度

ax2.legend(frameon=True, loc='upper left', markerscale=1.5, 
           prop={'size': 12, 'weight': 'bold'},
           facecolor='white', edgecolor='black', framealpha=1)

for i, ax in enumerate([ax1, ax2]):
    ax.set_xlim(_xlim)
    ax.set_ylim(_ylim)
    
    # 开启厚边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color('black')
    
    # 移除数值但保留刻度线（Tick marks）
    ax.set_xticks([])
    ax.set_yticks([]) # 如果 ax1 需要 y 轴刻度，可以单独设
    # 保持刻度向外
    ax.tick_params(which='both', direction='out', top=True, right=True, length=4)

    # 锁定比例但允许自动调整范围
    #ax.set_aspect('equal', adjustable='datalim')
    ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig(output_dir / "Figure_2_TSNE_cluster_activity.svg", format='svg', bbox_inches='tight')
plt.show()

# %%
# 12. Cluster Enrichment
print("\nCluster Activity Enrichment:")
enrichment = df.groupby('cluster_id')['label'].mean().sort_values(ascending=False)
print(enrichment)

# %%
# 13. Scaffold Analysis
from rdkit.Chem.Scaffolds import MurckoScaffold

def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    # Murcko can return "" for acyclic / peptide-like chains (no ring scaffold to strip)
    s = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    return s if s else None

df['scaffold'] = df['smiles'].apply(get_scaffold)

# %%
# 14. Scaffold Summary（加入活性比例）
scaffold_summary = (
    df.groupby(['cluster_id', 'scaffold'])
      .agg(
          count=('label', 'size'),
          active_ratio=('label', 'mean')
      )
      .reset_index()
)
# %%
# 15. Top scaffolds per cluster（Top 3）
top_scaffold = (
    scaffold_summary
    .sort_values(['cluster_id', 'count'], ascending=[True, False])
    .groupby('cluster_id')
    .head(3)
)

print("\nTop scaffolds per cluster:")
print(top_scaffold)


# %%
# 16. Representative Molecule Selection
from scipy.spatial.distance import cdist

representatives = []

for cid in sorted(df['cluster_id'].unique()):
    subset = df[df['cluster_id'] == cid]
    
    # 优先选 active
    subset_active = subset[subset['label'] == 1]
    if len(subset_active) > 0:
        subset = subset_active
    
    coords = subset[['tsne_1', 'tsne_2']].values
    center = coords.mean(axis=0)
    
    dists = cdist(coords, [center]).flatten()
    idx = np.argmin(dists)
    
    representatives.append(subset.iloc[idx])

rep_df = pd.DataFrame(representatives)

# %%
# 17. Draw Representative Molecules
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import io
from PIL import Image
def draw_highlighted_scaffold(mol_smiles, scaffold_smiles, legend, size=(400, 400)):
    mol = Chem.MolFromSmiles(mol_smiles)
    scaffold = Chem.MolFromSmiles(scaffold_smiles)
    
    if not mol or not scaffold:
        return None
    
    # 寻找骨架匹配
    params = Chem.SubstructMatchParameters()
    params.useChirality = False  # 忽略手性差异
    match = mol.GetSubstructMatch(scaffold, params)
    
    # 准备绘图对象
    drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    options = drawer.drawOptions()
    
    # --- 视觉样式优化 ---
    options.backgroundColour = (1, 1, 1, 1) # 纯白背景
    options.legendFontSize = 24
    options.annotationFontScale = 0.8
    options.bondLineWidth = 1.5
    options.fixedFontSize = 14
    
    # 高亮颜色设置 (使用淡蓝色半透明填充)
    highlight_color = (0.7, 0.85, 1.0, 0.6) 
    
    # 获取需要高亮的键
    hit_bonds = []
    for bond in scaffold.GetBonds():
        aid1 = match[bond.GetBeginAtomIdx()]
        aid2 = match[bond.GetEndAtomIdx()]
        hit_bonds.append(mol.GetBondBetweenAtoms(aid1, aid2).GetIdx())

    # 准备分子坐标
    mol = rdMolDraw2D.PrepareMolForDrawing(mol)
    
    # 执行绘制
    drawer.DrawMolecule(
        mol,
        legend=legend,
        highlightAtoms=match,
        highlightAtomColors={i: highlight_color for i in match},
        highlightBonds=hit_bonds,
        highlightBondColors={i: highlight_color for i in hit_bonds}
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()

# 批量生成 SVG
svg_images = []
for _, row in rep_df.iterrows():
    leg = f"Cluster {int(row['cluster_id'])} ({'Active' if row['label']==1 else 'Inactive'})"
    svg_text = draw_highlighted_scaffold(row['smiles'], row['scaffold'], leg)
    if svg_text:
        svg_images.append(svg_text)

# 将所有代表分子组合并保存
# 这里的处理方式是保存为独立的 SVG 序列，或拼接到一个 HTML 预览中
with open(output_dir / "Figure_3_scaffold_highlights.html", "w") as f:
    f.write("<html><body style='display: flex; flex-wrap: wrap;'>")
    for svg in svg_images:
        f.write(f"<div style='margin: 10px;'>{svg}</div>")
    f.write("</body></html>")

print(f"Highlighted molecules saved to {output_dir}")

from pathlib import Path

# 定义新文件夹路径
sep_svg_dir = output_dir / "representative_molecules_svg"
sep_svg_dir.mkdir(parents=True, exist_ok=True)

print(f"正在保存独立代表分子图至: {sep_svg_dir}")

for _, row in rep_df.iterrows():
    # 构造文件名：例如 Cluster_1_Active.svg
    status = 'Active' if row['label'] == 1 else 'Inactive'
    file_name = f"Cluster_{int(row['cluster_id'])}_{status}.svg"
    
    # 构造图例
    leg = f"Cluster {int(row['cluster_id'])} ({status})"
    
    # 调用之前定义的绘图函数获取 SVG 文本
    svg_text = draw_highlighted_scaffold(
        row['smiles'], 
        row['scaffold'], 
        leg, 
        size=(400, 400)
    )
    
    if svg_text:
        with open(sep_svg_dir / file_name, "w") as f:
            f.write(svg_text)

print(f"成功保存 {len(rep_df)} 个独立 SVG 文件。")
# %%
# 18. t-SNE Plot with Cluster + Enrichment (Optimized for Publication)
fig, ax = plt.subplots(figsize=(8, 7), dpi=600)

cluster_palette = sns.color_palette("Set2", manual_k)

for i, cid in enumerate(sorted(df['cluster_id'].unique())):
    subset = df[df['cluster_id'] == cid]
    ax.scatter(
        subset['tsne_1'], subset['tsne_2'],
        color=cluster_palette[i],
        s=45, alpha=0.7,
        edgecolors='white', linewidths=0.3,
        label=f'C{cid}',
        zorder=1
    )

# 标注优化：增加颜色深浅区分富集度
for cid in sorted(df['cluster_id'].unique()):
    subset = df[df['cluster_id'] == cid]
    x, y = subset['tsne_1'].mean(), subset['tsne_2'].mean()
    ratio = enrichment.loc[cid]
    
    # 根据活性比例自动调整框的颜色：高活性用浅蓝色背景突出
    box_color = '#E1F5FE' if ratio > df['label'].mean() else 'white'
    
    ax.text(
        x, y,
        f"C{cid}\n{ratio:.1%}", # 使用百分比展示更直观
        fontsize=10, fontweight='bold',
        ha='center', va='center',
        zorder=3,
        bbox=dict(
            boxstyle='round,pad=0.3', 
            fc=box_color, 
            ec='black', 
            lw=0.8, 
            alpha=0.9
        )
    )

# 细节美化
ax.set_title('Chemical Space & Bioactivity Enrichment', loc='left', fontweight='bold', pad=20)
ax.set_xlabel('t-SNE dimension 1')
ax.set_ylabel('t-SNE dimension 2')

# 延续你的全封闭箱式风格
for spine in ['top', 'right', 'left', 'bottom']:
    ax.spines[spine].set_visible(True)
    ax.spines[spine].set_linewidth(1.3)

ax.set_xticks([])
ax.set_yticks([])
ax.tick_params(direction='out', top=True, right=True, length=4)

ax.legend(
    title="Clusters", 
    loc='upper left', 
    bbox_to_anchor=(0.02, 0.98), # 微调位置，确保不贴死边框
    markerscale=1.4,
    prop={'size': 11, 'weight': 'bold'},
    frameon=True, 
    facecolor='white', 
    edgecolor='black', 
    framealpha=1, 
    handletextpad=0.1,
    fontsize=11,
    ncol=1 # 如果 C1-C8 太长，可以改为 ncol=2
)

plt.tight_layout()
plt.savefig(output_dir / "Figure_2_TSNE_Enrichment_Polished.svg", bbox_inches='tight')
plt.show()

# %% 19
import matplotlib.ticker as mtick

# 1. 重新准备数据，确保不被之前的排序干扰
# 直接按 cluster_id 从小到大排序，确保与坐标轴顺序一致
res_df = df.groupby('cluster_id')['label'].agg(['mean', 'size']).reset_index()
res_df.columns = ['cluster_id', 'active_ratio', 'count']
res_df = res_df.sort_values('cluster_id') 

avg_ratio = df['label'].mean() 

plt.figure(figsize=(10, 6), dpi=600)

# 使用 Set2 色板，确保与 t-SNE 颜色一致
cluster_palette = sns.color_palette("Set2", len(res_df))

ax = sns.barplot(
    x='cluster_id', 
    y='active_ratio', 
    data=res_df, 
    palette=cluster_palette,
    edgecolor='black',
    linewidth=1.2
)

# 2. 使用 enumerate 确保标注的是当前的 bar
for i, p in enumerate(ax.patches):
    # 此时 i 对应 res_df 的行索引
    row = res_df.iloc[i]
    ratio = row['active_ratio']
    n_total = int(row['count'])
    
    # 获取柱子中心位置
    x = p.get_x() + p.get_width() / 2.
    y = p.get_height()
    
    # 只在有高度的情况下标注（或 0% 也标注在底部）
    ax.text(x, y + 0.01, f'{ratio:.1%}\n(n={n_total})', 
            ha='center', va='bottom', 
            fontsize=10, fontweight='bold')

# 3. 辅助线与美化
plt.axhline(y=avg_ratio, color='#D62728', linestyle='--', linewidth=1.5, 
            label=f'Global Avg: {avg_ratio:.1%}')

# 4. 视觉优化与边框加固
plt.title('(c) NSD2 Bioactivity Enrichment per Cluster', loc='left', fontweight='bold', pad=25, fontsize=14)
plt.ylabel('Active Molecule Ratio (%)', fontsize=12, fontweight='bold')
plt.xlabel('Cluster ID', fontsize=12, fontweight='bold')
plt.ylim(0, 1.05) # 增加顶部空白空间

for spine in ['top', 'right', 'left', 'bottom']:
    ax.spines[spine].set_visible(True)
    ax.spines[spine].set_linewidth(1.5)
    ax.spines[spine].set_color('black')

# 格式化 Y 轴
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.tick_params(direction='in', length=6, width=1.2, top=False, right=False) # 刻度线

# 5. 图例
ax.legend(
    loc='upper center', 
    bbox_to_anchor=(0.5, 1.0), # 放在顶部中间靠下的位置
    ncol=1, 
    frameon=True, 
    facecolor='white', 
    edgecolor='black', 
    framealpha=1,
    prop={'size': 11, 'weight': 'bold'}
)

plt.tight_layout()
plt.savefig(output_dir / "Figure_S3_enrichment_bar_polished.svg", bbox_inches='tight')
plt.show()

# %% 19.1 Combined Final Figure (A, B, C)
import matplotlib.gridspec as gridspec

# 绘图风格全局微调
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Cambria', 'Times New Roman'],
    'axes.linewidth': 1.5,
    'savefig.dpi': 600
})

# 创建画布：设置 2 行 2 列，但第二行跨越所有列
fig = plt.figure(figsize=(14, 12), dpi=600)
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 0.7], hspace=0.25, wspace=0.15)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, :]) # 跨越整行

# --- (A) Structural Clusters ---
cluster_palette = sns.color_palette("Set2", manual_k)
for i in range(manual_k):
    subset = df[df['cluster'] == i]
    ax_a.scatter(subset['tsne_1'], subset['tsne_2'],
                 color=cluster_palette[i], s=55, alpha=0.9,
                 edgecolors='white', linewidths=0.3, label=f'C{i+1}')

ax_a.set_title('(A)', loc='left', fontweight='bold', fontsize=18, pad=10)
ax_a.set_xlabel('t-SNE dimension 1', fontsize=12)
ax_a.set_ylabel('t-SNE dimension 2', fontsize=12)
ax_a.legend(frameon=True, ncol=2, loc='upper left', facecolor='white', edgecolor='black', fontsize=10)

# --- (B) Bioactivity Distribution ---
activity_colors = {1: '#28559A', 0: '#BDBDBD'}
for val in [0, 1]:
    subset = df[df['label'] == val]
    ax_b.scatter(subset['tsne_1'], subset['tsne_2'],
                 color=activity_colors[val], s=55, alpha=1,
                 edgecolors='white', linewidths=0.3,
                 label='Active' if val == 1 else 'Inactive',
                 zorder=2 if val == 1 else 1)

ax_b.set_title('(B)', loc='left', fontweight='bold', fontsize=18, pad=10)
ax_b.set_xlabel('t-SNE dimension 1', fontsize=12)
ax_b.legend(frameon=True, loc='upper left', facecolor='white', edgecolor='black', fontsize=11)

# 统一 A 和 B 的比例与刻度
for ax in [ax_a, ax_b]:
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')

# --- (C) Enrichment Bar Chart ---
res_df = df.groupby('cluster_id')['label'].agg(['mean', 'size']).reset_index()
res_df.columns = ['cluster_id', 'active_ratio', 'count']
res_df = res_df.sort_values('cluster_id')

sns.barplot(x='cluster_id', y='active_ratio', data=res_df, 
            palette=cluster_palette, edgecolor='black', linewidth=1.2, ax=ax_c)

# 标注数值
for i, p in enumerate(ax_c.patches):
    row = res_df.iloc[i]
    ax_c.text(p.get_x() + p.get_width()/2., p.get_height() + 0.02,
              f'{row["active_ratio"]:.1%}\n(n={int(row["count"])})',
              ha='center', va='bottom', fontsize=10, fontweight='bold')

# 辅助线
avg_ratio = df['label'].mean()
ax_c.axhline(y=avg_ratio, color='#D62728', linestyle='--', linewidth=2, label=f'Global Avg: {avg_ratio:.1%}')

ax_c.set_title('(C)', loc='left', fontweight='bold', fontsize=18, pad=15)
ax_c.set_ylabel('Active Molecule Ratio (%)', fontsize=12, fontweight='bold')
ax_c.set_xlabel('Cluster ID', fontsize=12, fontweight='bold')
ax_c.set_ylim(0, 1.1)
ax_c.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax_c.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='black')

# 最终布局调整
plt.tight_layout()
plt.savefig(output_dir / "Figure_2_Combined_Analysis.svg", format='svg', bbox_inches='tight')
plt.show()

# %%
# 20. Save Key Tables
top_scaffold.to_csv(output_dir / "top_scaffolds.csv", index=False)
rep_df.to_csv(output_dir / "representative_molecules.csv", index=False)

print("\nAll outputs saved to:", output_dir)
# %%
