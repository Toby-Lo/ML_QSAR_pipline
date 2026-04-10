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
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'svg.fonttype': 'none',   # 保持字体可编辑
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
# 4. Morgan Fingerprints（新API）
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

# 期刊风
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(direction='out', length=5, width=1)

plt.tight_layout()
plt.savefig(output_dir / "Figure_S1_PCA_scatter.svg", bbox_inches='tight')
plt.show()

# %%
# 8. t-SNE（仅可视化）
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
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6))

cluster_palette = sns.color_palette("husl", manual_k)
activity_palette = {1: '#28559A', 0: '#B73131'}

# --- A. Clusters ---
for i in range(manual_k):
    subset = df[df['cluster'] == i]
    ax1.scatter(
        subset['tsne_1'], subset['tsne_2'],
        color=cluster_palette[i],
        s=25, alpha=0.6,
        label=f'Cluster {i+1}'
    )

ax1.set_title('(A) Chemical Space Clusters', fontweight='bold')
ax1.set_xlabel('t-SNE 1')
ax1.set_ylabel('t-SNE 2')
ax1.legend(frameon=True, fontsize=9)

# --- B. Activity ---
for val in [0,1]:
    subset = df[df['label']==val]
    ax2.scatter(
        subset['tsne_1'], subset['tsne_2'],
        color=activity_palette[val],
        s=25, alpha=0.6,
        label='Active' if val==1 else 'Inactive'
    )

ax2.set_title('(B) Bioactivity Distribution', fontweight='bold')
ax2.set_xlabel('t-SNE 1')
ax2.legend(frameon=True)

# --- Style ---
for ax in [ax1, ax2]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    ax.tick_params(direction='out', length=5, width=1)

plt.tight_layout()
plt.savefig(output_dir / "Figure_2_TSNE_cluster_activity.svg", bbox_inches='tight')
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
    if mol:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    return None

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
# 16. Representative Molecule Selection（核心步骤）
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

mols = [Chem.MolFromSmiles(s) for s in rep_df['smiles']]

legends = [
    f"C{row['cluster_id']} | {'Active' if row['label']==1 else 'Inactive'}"
    for _, row in rep_df.iterrows()
]

img = Draw.MolsToGridImage(
    mols,
    molsPerRow=4,
    subImgSize=(300, 300),
    legends=legends,
    useSVG=True
)

with open(output_dir / "Figure_3_representative_molecules.svg", "w") as f:
    f.write(img.data)

# %%
# 18. t-SNE Plot with Cluster + Enrichment Annotation
fig, ax = plt.subplots(figsize=(7,6))

cluster_palette = sns.color_palette("husl", len(df['cluster_id'].unique()))

for i, cid in enumerate(sorted(df['cluster_id'].unique())):
    subset = df[df['cluster_id'] == cid]
    
    ax.scatter(
        subset['tsne_1'], subset['tsne_2'],
        color=cluster_palette[i],
        s=25, alpha=0.6,
        label=f'C{cid}'
    )

# 添加 enrichment annotation（关键）
for cid in df['cluster_id'].unique():
    subset = df[df['cluster_id'] == cid]
    
    x = subset['tsne_1'].mean()
    y = subset['tsne_2'].mean()
    
    ratio = enrichment.loc[cid]
    
    ax.text(
        x, y,
        f"C{cid}\n({ratio:.2f})",
        fontsize=9,
        ha='center',
        va='center',
        bbox=dict(boxstyle='round,pad=0.25', fc='white', alpha=0.7)
    )

ax.set_title('t-SNE Chemical Space with Cluster Enrichment', fontweight='bold')
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')

ax.legend(frameon=True, fontsize=8)

# 学术风格
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(direction='out', length=5, width=1)

plt.tight_layout()
plt.savefig(output_dir / "Figure_2_TSNE_enrichment.svg", bbox_inches='tight')
plt.show()

# %%
# 19. Cluster Enrichment Bar Plot
enrichment_df = enrichment.reset_index()

plt.figure(figsize=(6,4))
sns.barplot(x='cluster_id', y='label', data=enrichment_df)

plt.ylabel('Active Ratio')
plt.xlabel('Cluster ID')
plt.title('Cluster Activity Enrichment')

# 学术风格
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(direction='out', length=5, width=1)

plt.tight_layout()
plt.savefig(output_dir / "Figure_S3_enrichment_bar.svg", bbox_inches='tight')
plt.show()

# %%
# 20. Save Key Tables
top_scaffold.to_csv(output_dir / "top_scaffolds.csv", index=False)
rep_df.to_csv(output_dir / "representative_molecules.csv", index=False)

print("\nAll outputs saved to:", output_dir)
# %%
