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

act_path = "../data/NSD2/NSD2_chembl_act_2124.csv"
com_path = "../data/NSD2/NSD2_chembl_com_1265.csv"

# Check if files exist locally, if not, try the Data folder
'''
if not os.path.exists(act_path):
    act_path = "../data/NSD2/NSD2_chembl_act_2124.csv"
    com_path = "../data/NSD2/NSD2_chembl_com_1265.csv"
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

# 6. Label activity #### Adjust Threshold if necessary
final_df['label'] = np.where(final_df['pIC50'] >= 6.0, '1', '0') ##### 1 for Active, 0 for Inactive

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
final_df.to_csv("../data/NSD2/nsd2_final_dataset.csv", index=False) ###

print("Properties calculation complete.")
print(f'Final dataset shape: {final_df.shape}')

final_df.head(5)

# %%
# ### 5. Visualization 
from matplotlib.lines import Line2D
final_df['label'] = final_df['label'].astype(int)
final_df['label_name'] = final_df['label'].map({1: 'Active', 0: 'Inactive'})
# Update Global Matplotlib RC params for Top Journal Style
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],  # 'Cambria'
    'font.size': 11,

    'axes.linewidth': 1.0,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.grid': False,

    'xtick.direction': 'out', #刻度线朝向
    'ytick.direction': 'out',

    'savefig.dpi': 600,
    'svg.fonttype': 'none'
})

fig = plt.figure(figsize=(12.8, 4.5))
gs = mpl.gridspec.GridSpec(1, 3, width_ratios=[1, 0.8, 1.2])

color_palette = {'Active': '#28559A', 'Inactive': '#B73131'}

# --- (a) pIC50 Distribution ---
ax0 = fig.add_subplot(gs[0])

sns.histplot(
    final_df['pIC50'],
    bins=25,
    stat='density',
    color=color_palette['Active'],
    alpha=0.7,
    edgecolor=None,
    ax=ax0
)

# threshold line color and value ################
ax0.axvline(6.0, color='#D62728', linestyle='--', linewidth=1.4)

# legend
threshold_handle = Line2D(
    [0], [0],
    color='#D62728',
    linestyle='--',
    linewidth=1.4,
    label='Threshold:6.0'
)

ax0.legend(handles=[threshold_handle], frameon=True, loc='upper right')

ax0.set_title('pIC50 Distribution', fontweight='bold', pad=12)
ax0.set_xlabel('pIC50')
ax0.set_ylabel('Density')
ax0.xaxis.set_major_locator(MultipleLocator(1))

# --- (b) Class Balance ---
ax1 = fig.add_subplot(gs[1])

counts = final_df['label_name'].value_counts().reindex(['Active', 'Inactive'])
total = counts.sum()

sns.barplot(
    x=counts.index,
    y=counts.values,
    palette=color_palette,
    ax=ax1
)

for i, count in enumerate(counts):
    ax1.text(
        i,
        count * 1.02,
        f'{count}\n({100*count/total:.1f}%)',
        ha='center',
        va='bottom',
        fontsize=9
    )

ax1.set_title('Classification', fontweight='bold', pad=12)
ax1.set_xlabel('')
ax1.set_ylabel('Molecule Count')

# --- (c) Property Space (MW vs AlogP) ---
ax2 = fig.add_subplot(gs[2])

sns.scatterplot(
    x='MW',
    y='AlogP',
    hue='label_name',
    data=final_df,
    palette=color_palette,
    hue_order=['Active', 'Inactive'],
    s=25,
    alpha=0.6,
    edgecolor=None,
    ax=ax2,
    legend=False
)

# Lipinski 阈值也改为红色虚线
ax2.axvline(500, color='#D62728', linestyle='--', linewidth=1.2)
ax2.axhline(5, color='#D62728', linestyle=':', linewidth=1.2)

# custom legend（统一在一个框里）
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Active',
           markerfacecolor=color_palette['Active'], markersize=6),

    Line2D([0], [0], marker='o', color='w', label='Inactive',
           markerfacecolor=color_palette['Inactive'], markersize=6),

    Line2D([0], [0], color='#D62728', lw=1.2, linestyle='--',
           label='MW = 500'),

    Line2D([0], [0], color='#D62728', lw=1.2, linestyle=':',
           label='AlogP = 5'),
]

ax2.legend(
    handles=legend_elements,
    frameon=True,
    loc='lower right',
    #title='Annotations'
)

ax2.set_title('Chemical Property Space', fontweight='bold', pad=12)
ax2.set_xlabel('Molecular Weight (Da)')
ax2.set_ylabel('AlogP')

for n, ax in enumerate([ax0, ax1, ax2]):
    ax.text(-0.1, 1.05, f'({chr(97+n)})',
            transform=ax.transAxes,
            fontsize=13, fontweight='bold')
    
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(1.2)
        ax.spines[spine].set_color('black')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.tick_params(
        axis='both',
        which='both',
        direction='out',   # 向外
        length=4,          # 刻度长度
        width=1.0,         # 刻度粗细
        colors='black',    # 颜色
        top=False,
        right=False
    )
plt.tight_layout()

###################### export
output_dir = Path("../data/NSD2/")
output_dir.mkdir(parents=True, exist_ok=True)

fig.savefig(output_dir / "NSD2_EDA.png", bbox_inches='tight')
fig.savefig(output_dir / "NSD2_EDA.svg", bbox_inches='tight')
######################

plt.show()
print(f"EDA summary plots saved in {output_dir}")

# %%
