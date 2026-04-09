# %% 
# ### 1. Import Libraries and Configuration
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from pathlib import Path
import os

# Set visualization style
%matplotlib inline
sns.set_theme(style="whitegrid")

# Helper function to compute descriptors
def _compute_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return {
            'MW': Descriptors.MolWt(mol),
            'AlogP': Descriptors.MolLogP(mol),
            'HBD': Descriptors.NumHDonors(mol),
            'HBA': Descriptors.NumHAcceptors(mol)
        }
    return {'MW': None, 'AlogP': None, 'HBD': None, 'HBA': None}

# Function to clean SMILES (remove salts, canonicalize)
def clean_smiles(smiles):
    if pd.isna(smiles): return None
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    frags = Chem.GetMolFrags(mol, asMols=True)
    largest = max(frags, key=lambda x: x.GetNumHeavyAtoms())
    return Chem.MolToSmiles(largest, isomericSmiles=True)

print("Setup complete. Functions defined.")

# %%
# ### 2. Read and Merge Data
# Adjust paths if necessary

act_path = "../Data/NSD2/NSD2_chembl_act_2124.csv"
com_path = "../Data/NSD2/NSD2_chembl_com_1265.csv"

# Check if files exist locally, if not, try the Data folder
'''
if not os.path.exists(act_path):
    act_path = "../Data/NSD2/NSD2_chembl_act_2124.csv"
    com_path = "../Data/NSD2/NSD2_chembl_com_1265.csv"
'''

df_act = pd.read_csv(act_path)
df_com = pd.read_csv(com_path)

# Unified column names to lowercase
df_act.columns = [c.lower().replace(' ', '_') for c in df_act.columns]
df_com.columns = [c.lower().replace(' ', '_') for c in df_com.columns]

if 'chembl_id' in df_com.columns:
    df_com = df_com.rename(columns={'chembl_id': 'molecule_chembl_id'})

# Inner merge on molecule_chembl_id
merged = df_act.merge(df_com, on='molecule_chembl_id', how='inner', suffixes=('', '_drop'))
merged = merged.drop(columns=[c for c in merged.columns if c.endswith('_drop')])

print(f"Merged Data: {merged.shape}")
merged.head(2)

# %%
# ### 3. Bioactivity Filtering and SMILES Cleaning

# 1. Filter IC50 and nM units
df_ic50 = merged[(merged['standard_type'].str.upper() == 'IC50') & 
                 (merged['standard_units'].str.lower() == 'nm') &
                 (merged['standard_relation'] == "'='")
].copy()

# 2. Convert value to numeric and drop NAs
df_ic50['standard_value'] = pd.to_numeric(df_ic50['standard_value'], errors='coerce')
df_ic50 = df_ic50.dropna(subset=['standard_value'])

# 3. Calculate pIC50
df_ic50['pIC50'] = -np.log10(df_ic50['standard_value'] * 1e-9)
# 过滤掉 pIC50 小于 3 的分子 
# 1mM (pIC50=3) 已经是药物筛选中极低活性的极限了，低于这个值的分子没有统计意义
df_ic50 = df_ic50[df_ic50['pIC50'] >= 3.0]

# 4. Canonicalize SMILES and remove salts
print("Cleaning SMILES (Removing salts)...")
df_ic50['canonical_smiles'] = df_ic50['smiles'].apply(clean_smiles)
df_ic50 = df_ic50.dropna(subset=['canonical_smiles'])

# 5. Aggregate duplicates by median pIC50
final_df = df_ic50.groupby('canonical_smiles').agg({
    'molecule_chembl_id': 'first',
    'pIC50': 'median',
    'standard_value': 'median'
}).reset_index()

# 6. Label activity
final_df['label'] = np.where(final_df['pIC50'] >= 6.0, 'Active', 'Inactive') ###

print(f"Unique Molecules after cleaning: {len(final_df)}")

# %%
# ### 4. Calculate Molecular Properties (MW & AlogP)

print("Calculating Physicochemical Properties...")
# Drop existing MW/AlogP if they exist to avoid conflict
for col in ['MW', 'AlogP']:
    if col in final_df.columns:
        final_df = final_df.drop(columns=[col])

# Apply RDKit descriptors
df_props = final_df['canonical_smiles'].apply(lambda x: pd.Series(_compute_properties(x)))
final_df = pd.concat([final_df.reset_index(drop=True), df_props], axis=1)
# save
final_df.to_csv("../Data/JAK3/JAK3_final_dataset.csv", index=False)

print("Properties calculation complete.")
print(f'Final dataset shape: {final_df.shape}')
#print(f'Final csv has been saved to ../Data/JAK3/JAK3_final_dataset.csv')
final_df.head(5)

# %%
# ### 5. Visualization 
# Update Global Matplotlib RC params for Top Journal Style
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,

    # 坐标轴线配置
    'axes.linewidth': 1.0,       # 轴线粗细
    'axes.spines.right': False,  # 隐藏右边框
    'axes.spines.top': False,    # 隐藏上边框
    'axes.grid': False,          # 彻底关闭网格

    'xtick.direction': 'in',     # 刻度向内
    'ytick.direction': 'in',
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'savefig.dpi': 600,
    'svg.fonttype': 'none'
})

fig = plt.figure(figsize=(13, 4.5)) 
gs = mpl.gridspec.GridSpec(1, 3, width_ratios=[1, 0.8, 1.2])

# 调色盘：深蓝 vs 深红
color_palette = {'Active': '#28559A', 'Inactive': '#B73131'}

# --- (a) pIC50 Distribution ---
ax0 = fig.add_subplot(gs[0])
sns.histplot(final_df['pIC50'], bins=25, kde=True, ax=ax0, 
             color=color_palette['Active'], alpha=0.6, edgecolor='white', linewidth=0.5)
# 阈值改为红色################
ax0.axvline(x=6.0, color='#D62728', linestyle='--', linewidth=1.5, label='Threshold')
ax0.set_title('pIC50 Distribution', fontweight='bold', pad=15)
ax0.set_xlabel('pIC50 Value')
ax0.set_ylabel('Count')
ax0.xaxis.set_major_locator(MultipleLocator(1))
ax0.legend(frameon=False, loc='upper right')

# --- (b) Class Balance ---
ax1 = fig.add_subplot(gs[1])
counts = final_df['label'].value_counts()
total = counts.sum()
sns.countplot(x='label', data=final_df, order=['Active', 'Inactive'], ax=ax1, 
              hue='label', palette=color_palette, legend=False, alpha=0.85, edgecolor='black', linewidth=1.2)

for i, count in enumerate(counts.reindex(['Active', 'Inactive'])):
    ax1.text(i, count + (total*0.03), f'{100*count/total:.1f}%', ha='center', va='bottom', 
             fontsize=10, fontweight='bold')

ax1.set_title('Classification', fontweight='bold', pad=15)
ax1.set_xlabel('Biological Activity')
ax1.set_ylabel('Molecule Count')
ax1.set_ylim(0, total * 1.15)

# --- (c) Property Space (MW vs AlogP) ---
ax2 = fig.add_subplot(gs[2])
sns.scatterplot(x='MW', y='AlogP', hue='label', data=final_df,
                hue_order=['Active', 'Inactive'], palette=color_palette,
                ax=ax2, s=45, alpha=0.7, edgecolor='white', linewidth=0.6)

# Lipinski 阈值也改为红色虚线
ax2.axvline(x=500, color='#D62728', linestyle=':', linewidth=1.2, label='MW = 500')
ax2.axhline(y=5, color='#D62728', linestyle='-.', linewidth=1.2, label='AlogP = 5')

ax2.set_title('Chemical Property Space', fontweight='bold', pad=15)
ax2.set_xlabel('Molecular Weight (Da)')
ax2.set_ylabel('Calculated AlogP')
ax2.legend(frameon=False, loc='upper left', bbox_to_anchor=(1, 1), title='Activity')

# 添加子图标签 (a), (b), (c)
for n, ax in enumerate([ax0, ax1, ax2]):
    ax.text(-0.12, 1.12, f'({chr(97+n)})', transform=ax.transAxes, 
            fontsize=14, fontweight='bold', va='top')
    # 确保坐标轴线清晰展示
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

plt.tight_layout()

# export
output_dir = Path("../Data/NSD2/")
output_dir.mkdir(parents=True, exist_ok=True)
######################
fig.savefig(output_dir / "NSD2_EDA.png")#########
fig.savefig(output_dir / "NSD2_EDA.svg")########
######################

plt.show()
print(f"EDA summary plots saved in {output_dir}")

# %%
