import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans

# === Global Font Style ===
matplotlib.rcParams['font.family'] = 'serif'

# === Input/Output Settings ===
pca_file = "projection.txt"
output_fig = "pca.png"

# === Load PCA Data ===
data = np.loadtxt(pca_file)
pc1 = data[:, 1]
pc2 = data[:, 2]
pca_coords = np.vstack((pc1, pc2)).T
total_frames = len(pc1)

# === User Input ===
try:
    n_clusters = int(input("Enter number of clusters: "))
    total_ns = float(input("Enter total simulation time in nanoseconds: "))
except:
    print("Invalid input. Exiting.")
    exit()

time_per_frame = total_ns / total_frames

# === Density Estimation ===
xy = np.vstack([pc1, pc2])
z = gaussian_kde(xy)(xy)

# === KMeans Clustering ===
kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(pca_coords)
labels = kmeans.labels_

# === Setup ===
colors = ['red', 'orange', 'green', 'blue', 'purple', 'brown', 'cyan', 'magenta', 'darkgreen', 'gold']
label_names_all = [chr(i) for i in range(65, 91)]  # A–Z

# === Find Min-Energy Frames per Cluster ===
cluster_min_frames = []
for cluster_id in range(n_clusters):
    indices = np.where(labels == cluster_id)[0]
    cluster_z = z[indices]
    min_frame = indices[np.argmax(cluster_z)]
    cluster_min_frames.append((min_frame, cluster_id))

# === Sort Clusters by Simulation Time (Frame Number) ===
cluster_min_frames.sort()

# === Plot PCA with Labels and Arrows ===
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(pc1, pc2, c=z, s=15, cmap='viridis')

for tick in ax.get_xticklabels() + ax.get_yticklabels():
    tick.set_fontweight('bold')
    tick.set_fontname('serif')

arrow_coords = []

# === Annotate Cluster Centers ===
for i, (frame_index, cluster_id) in enumerate(cluster_min_frames):
    x = pc1[frame_index]
    y = pc2[frame_index]
    ns_time = frame_index * time_per_frame
    label = label_names_all[i]
    color = colors[i % len(colors)]
    label_text = f"{label} ({ns_time:.1f} ns)"

    ax.scatter(x, y, color=color, s=80, zorder=3, edgecolor='black', linewidth=1.2)
    ax.annotate(label_text, (x, y), textcoords="offset points", xytext=(0, 12),
                ha='center', color=color, fontweight='bold', fontsize=12, family='serif')

    print(f"Cluster {label}: Time {ns_time:.1f} ns, Color: {color}")
    arrow_coords.append((x, y))

# === Draw Curved Light Gray Arrows Between Clusters ===
for i in range(len(arrow_coords) - 1):
    (x_start, y_start), (x_end, y_end) = arrow_coords[i], arrow_coords[i + 1]
    arrow = FancyArrowPatch(
        (x_start, y_start), (x_end, y_end),
        connectionstyle="arc3,rad=0.2",
        arrowstyle='-|>',
        color='gray',
        linewidth=2,
        mutation_scale=20,
        zorder=2
    )
    ax.add_patch(arrow)

# === Finalize and Save ===
ax.set_xlabel('PC1', fontsize=14, fontweight='bold')
ax.set_ylabel('PC2', fontsize=14, fontweight='bold')
ax.tick_params(labelsize=12)
plt.tight_layout()
plt.savefig(output_fig, dpi=600)
print(f"\nPCA plot saved as '{output_fig}' with elegant gray arrows and Times New Roman font.")
