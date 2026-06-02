#!/usr/bin/env python3
"""Plot PC1 vs PC2 with a Morandi-style time gradient.

Color meaning:
  - Red:   early simulation
  - Purple: middle transition
  - Blue:  late simulation

Example:
  python3 ../../Analysis/14_PCA/plot_pca_morandi_timecolored.py \
    --input analysis/PCA_projection.dat \
    --out analysis/plots/14_PCA_PC1_PC2_morandi_timecolored.svg \
    --total-ns 200

for d in mds/runs/*; do \
  [ -d "$d" ] || continue; \
  echo "=== Drawing for $d ==="; \
  python3 mds/Analysis/14_PCA/plot_pca_morandi_timecolored.py \
    --input "$d/analysis/PCA_projection.dat" \
    --out "$d/analysis/plots/14_PCA_PC1_PC2_morandi_timecolored.svg" \
    --total-ns 200; \
done
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import rcParams


def set_publication_style() -> None:
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = ["Cambria", "Times New Roman", "DejaVu Serif"]
    rcParams["mathtext.fontset"] = "stix"
    rcParams["font.size"] = 10.5
    rcParams["font.weight"] = "bold"
    rcParams["axes.labelweight"] = "bold"
    rcParams["axes.titleweight"] = "bold"
    rcParams["axes.linewidth"] = 1.0
    rcParams["xtick.direction"] = "out"
    rcParams["ytick.direction"] = "out"
    rcParams["xtick.major.width"] = 1.0
    rcParams["ytick.major.width"] = 1.0
    rcParams["xtick.major.size"] = 4.0
    rcParams["ytick.major.size"] = 4.0
    rcParams["xtick.top"] = False
    rcParams["ytick.right"] = False


def morandi_time_cmap() -> LinearSegmentedColormap:
    # Journal-style yellow -> deep purple gradient inspired by Viridis/Plasma.
    colors = ["#FDE725", "#7E03A8", "#440154"]
    return LinearSegmentedColormap.from_list("morandi_time", colors, N=256)


def load_projection(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments=["#", "@"])
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        raise SystemExit(f"Need at least frame+PC1+PC2 in {path}, got shape={data.shape}")
    frames = data[:, 0].astype(float)
    pc1 = data[:, 1].astype(float)
    pc2 = data[:, 2].astype(float)
    return frames, pc1, pc2


def frames_to_time_ns(frames: np.ndarray, total_ns: float) -> np.ndarray:
    if frames.size <= 1:
        return np.zeros_like(frames, dtype=float)
    order = np.argsort(frames)
    time_ns = np.empty_like(frames, dtype=float)
    time_ns[order] = np.linspace(0.0, total_ns, frames.size)
    return time_ns


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot PCA projection with a Morandi time gradient.")
    ap.add_argument("-i", "--input", type=Path, default=Path("analysis/PCA_projection.dat"))
    ap.add_argument("-o", "--out", type=Path, default=Path("analysis/plots/14_PCA_PC1_PC2_morandi_timecolored.svg"))
    ap.add_argument("--total-ns", type=float, default=200.0, help="Total simulation time in ns.")
    ap.add_argument("--title", default="PC1-PC2 Time-Colored Projection")
    args = ap.parse_args()

    set_publication_style()
    frames, pc1, pc2 = load_projection(args.input)
    time_ns = frames_to_time_ns(frames, args.total_ns)

    cmap = morandi_time_cmap()
    norm = Normalize(vmin=0.0, vmax=float(args.total_ns), clip=True)
    draw_order = np.argsort(time_ns)[::-1]  # Late frames first so early colors stay visible on top.

    fig, ax = plt.subplots(figsize=(6.5, 5.4), facecolor="white")
    ax.set_facecolor("white")

    sc = ax.scatter(
        pc1[draw_order],
        pc2[draw_order],
        c=time_ns[draw_order],
        cmap=cmap,
        norm=norm,
        s=12,
        alpha=0.92,
        linewidths=0.0,
    )

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Time (ns)")
    cbar.set_ticks(np.linspace(0.0, float(args.total_ns), 5))

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(args.title, fontweight="bold")
    ax.tick_params(axis="both", which="major", direction="out", width=1.0, length=4)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    cbar.ax.tick_params(direction="out", width=1.0, length=4)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight("bold")
    cbar.ax.yaxis.label.set_fontweight("bold")

    ax.grid(False)
    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=600, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)

    print(f"[OK] Saved: {args.out.resolve()}")


if __name__ == "__main__":
    main()
