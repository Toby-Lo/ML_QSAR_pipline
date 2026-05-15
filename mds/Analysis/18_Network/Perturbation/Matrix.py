import numpy as np
import matplotlib.pyplot as plt

# Load covariance matrix
cov = np.loadtxt("corr_mat.txt")
n_residues = cov.shape[0] // 3

# Compute per-residue covariance by averaging 3x3 blocks
res_cov = np.zeros((n_residues, n_residues))
for i in range(n_residues):
    for j in range(n_residues):
        block = cov[i*3:(i+1)*3, j*3:(j+1)*3]
        res_cov[i, j] = np.mean(block)

# Convert to correlation matrix
std = np.sqrt(np.diag(res_cov))
corr = res_cov / np.outer(std, std)
corr[np.isnan(corr)] = 0  # Handle divide-by-zero if any

# Set global font to serif and bold
plt.rcParams.update({
    "font.family": "serif",
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titlesize": "large",
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12
})

# Plot
plt.figure(figsize=(10, 8))
img = plt.imshow(corr, cmap='bwr', origin='lower', vmin=-1, vmax=1)
cbar = plt.colorbar(img)
cbar.set_label('Correlation', weight='bold', fontsize=14)
plt.xlabel('Residue Index')
plt.ylabel('Residue Index')
plt.tight_layout()
plt.savefig("PRS_residue_correlation.png", dpi=300)
plt.show()
