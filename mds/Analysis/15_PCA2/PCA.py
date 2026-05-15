import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
import sys

# Set global font to Times New Roman
matplotlib.rcParams['font.family'] = ['Times New Roman', 'Times', 'serif']

# Load the data
a = np.loadtxt(sys.argv[1])
pc1 = a[:, 1]
pc2 = a[:, 2]

# Generate density estimates
xy = np.vstack([pc1, pc2])
z = gaussian_kde(xy)(xy)

# Create the plot
fig, ax = plt.subplots()
ax.scatter(pc1, pc2, c=z, s=5)
for label in ax.get_xticklabels():
    label.set_fontweight('bold')
for label in ax.get_yticklabels():
    label.set_fontweight('bold')
# Define the mapping between frame numbers and labels
frame_to_label = {
#    756: 'A-11%',
    1777: 'A',
    5558: 'B',
    8608: 'C'  # Add or modify as necessary
}

# Annotate specific frame numbers
for frame_number, label in frame_to_label.items():
    if frame_number < len(pc1):  # Ensure the frame number is within range
        x = pc1[frame_number]
        y = pc2[frame_number]
        ax.scatter(x, y, color='red')  # Circle the point in red
        ax.annotate(label, (x, y),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center',
                    color='red',
                    fontweight='bold')  # Red, bold text

# Label the axes and the plot
#plt.xlabel('pc1', fontsize=14, fontweight='bold')
#plt.ylabel('pc2', fontsize=14, fontweight='bold')
#plt.title(sys.argv[3], fontsize=20)

# Save and show the plot
plt.savefig(sys.argv[2])
plt.show()
