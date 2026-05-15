#!/usr/bin/env python3
"""
Publication-style PCA + FEL figures.

Outputs:
  1) PCA variance contribution bar chart (+ cumulative line)
  2) PC1 vs PC2 time-colored projection scatter
  3) FEL contour map from PC1/PC2 probability density via
     DeltaG = -RT ln(P/Pmax)   [kcal/mol]
  4) Optional 3D FEL surface map

Example:
  python3 ../../Analysis/16_FEL/pca_fel_publication_plots.py \
    --projection analysis/PCA_projection.dat \
    --outdir analysis/plots \
    --dt 0.002 --ntwx 5000 --stride 10 --temperature 300
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

R_KCAL = 0.0019872041  # kcal/(mol*K)


def set_publication_style() -> None:
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = ["Cambria", "Times New Roman", "DejaVu Serif"]
    rcParams["mathtext.fontset"] = "stix"
    rcParams["font.size"] = 10.5
    rcParams["axes.linewidth"] = 1.0
    rcParams["xtick.direction"] = "in"
    rcParams["ytick.direction"] = "in"
    rcParams["xtick.top"] = False
    rcParams["ytick.right"] = False


def load_projection(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments=["#", "@"])  # frame + PCs
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        raise SystemExit(f"Need at least frame+PC1+PC2 in {path}, got shape={data.shape}")
    frames = data[:, 0].astype(float)
    pcs = data[:, 1:].astype(float)
    return frames, pcs


def frames_to_ns(frames: np.ndarray, dt: float, ntwx: int, stride: int) -> np.ndarray:
    return (frames * ntwx * stride * dt) / 1000.0


def explained_variance_ratio(pcs: np.ndarray) -> np.ndarray:
    # Variance of each projected PC. For orthogonal PCs from covariance eigendecomp,
    # these are proportional to eigenvalues.
    var = np.var(pcs, axis=0, ddof=1)
    total = np.sum(var)
    if total <= 0:
        return np.zeros_like(var)
    return var / total


def plot_variance(ratio: np.ndarray, out: Path, top_n: int) -> None:
    n = min(top_n, ratio.size)
    idx = np.arange(1, n + 1)
    pct = ratio[:n] * 100.0
    cum = np.cumsum(pct)

    fig, ax1 = plt.subplots(figsize=(6.8, 4.2), facecolor="white")
    ax1.set_facecolor("white")

    ax1.bar(idx, pct, color="#4c78a8", alpha=0.9, width=0.72, edgecolor="black", linewidth=0.4)
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance (%)")
    ax1.set_xticks(idx)
    ax1.set_xticklabels([f"PC{i}" for i in idx])
    ax1.set_xlim(0.4, n + 0.6)
    ax1.set_ylim(0, max(np.max(pct) * 1.25, 5))
    ax1.grid(False)

    ax2 = ax1.twinx()
    ax2.plot(idx, cum, color="#e45756", marker="o", linewidth=1.8, markersize=4.5)
    ax2.set_ylabel("Cumulative Variance (%)")
    ax2.set_ylim(0, 100)

    for x, y in zip(idx, cum):
        ax2.text(x, y + 1.5, f"{y:.1f}%", ha="center", va="bottom", fontsize=8)

    ax1.set_title("PCA Variance Contribution")
    plt.tight_layout()
    plt.savefig(out, dpi=600, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)


def plot_pc12_scatter(pc1: np.ndarray, pc2: np.ndarray, t_ns: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.3, 5.2), facecolor="white")
    ax.set_facecolor("white")

    sc = ax.scatter(pc1, pc2, c=t_ns, cmap="turbo", s=10, alpha=0.88, linewidths=0)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Time (ns)")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Trajectory Projection on PC1-PC2")
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(out, dpi=600, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)


def compute_fel(pc1: np.ndarray, pc2: np.ndarray, temperature: float, bins: int = 120):
    H, xedges, yedges = np.histogram2d(pc1, pc2, bins=bins, density=True)
    P = H.T  # shape: y, x

    # Smooth probability density to avoid grainy FEL from sparse sampling.
    P = gaussian_smooth2d(P)

    with np.errstate(divide="ignore", invalid="ignore"):
        Pmax = np.nanmax(P)
        G = -R_KCAL * temperature * np.log(P / Pmax)

    G[~np.isfinite(G)] = np.nan
    # Shift minimum to 0 for readability
    gmin = np.nanmin(G)
    if np.isfinite(gmin):
        G = G - gmin

    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    X, Y = np.meshgrid(xc, yc)
    return X, Y, G


def gaussian_kernel1d(sigma: float, radius: int) -> np.ndarray:
    x = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-(x ** 2) / (2.0 * sigma * sigma))
    k /= np.sum(k)
    return k


def gaussian_smooth2d(arr: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    if sigma <= 0:
        return arr
    radius = max(1, int(np.ceil(3.0 * sigma)))
    k = gaussian_kernel1d(sigma, radius)

    # Convolve along x then y (separable Gaussian).
    tmp = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), axis=1, arr=arr)
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), axis=0, arr=tmp)
    return out


def plot_fel_contour(X: np.ndarray, Y: np.ndarray, G: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.3), facecolor="white")
    ax.set_facecolor("white")

    vmax = np.nanpercentile(G, 97)
    vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
    levels = np.linspace(0, vmax, 24)
    cf = ax.contourf(X, Y, G, levels=levels, cmap="viridis")
    ax.contour(X, Y, G, levels=levels[::3], colors="white", linewidths=0.35, alpha=0.55)

    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(r"$\Delta G$ (kcal/mol)")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Free Energy Landscape (PC1-PC2)")
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(out, dpi=600, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)


def plot_fel_surface(X: np.ndarray, Y: np.ndarray, G: np.ndarray, out: Path) -> None:
    fig = plt.figure(figsize=(7.2, 5.8), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")

    # Clip extreme NaN/inf regions for cleaner surface
    Gp = np.array(G, copy=True)
    vmax = np.nanpercentile(Gp, 95)
    Gp = np.where(np.isfinite(Gp), np.minimum(Gp, vmax), np.nan)

    surf = ax.plot_surface(X, Y, Gp, cmap="viridis", linewidth=0, antialiased=True, alpha=0.98)
    cbar = fig.colorbar(surf, ax=ax, shrink=0.65, pad=0.08)
    cbar.set_label(r"$\Delta G$ (kcal/mol)")

    ax.set_xlabel("PC1", labelpad=8)
    ax.set_ylabel("PC2", labelpad=8)
    ax.set_zlabel(r"$\Delta G$ (kcal/mol)", labelpad=8)
    ax.set_title("Free Energy Landscape (3D Surface)")

    plt.tight_layout()
    plt.savefig(out, dpi=600, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate publication-style PCA + FEL figures.")
    ap.add_argument("--projection", type=Path, default=Path("analysis/PCA_projection.dat"))
    ap.add_argument("--outdir", type=Path, default=Path("analysis/plots"))
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--ntwx", type=int, default=5000)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--bins", type=int, default=0, help="2D histogram bins for FEL. 0 means auto.")
    ap.add_argument("--top-pc", type=int, default=6)
    args = ap.parse_args()

    set_publication_style()
    args.outdir.mkdir(parents=True, exist_ok=True)

    frames, pcs = load_projection(args.projection)
    t_ns = frames_to_ns(frames, args.dt, args.ntwx, args.stride)

    ratio = explained_variance_ratio(pcs)
    plot_variance(ratio, args.outdir / "14_PCA_variance_contribution.svg", args.top_pc)

    pc1 = pcs[:, 0]
    pc2 = pcs[:, 1]
    plot_pc12_scatter(pc1, pc2, t_ns, args.outdir / "14_PCA_PC1_PC2_timecolored.svg")

    bins = args.bins
    if bins <= 0:
        # Auto bins: enough detail for dense trajectories, but avoid overly sparse maps.
        n = len(pc1)
        bins = int(np.clip(np.sqrt(n) * 1.7, 60, 140))
    X, Y, G = compute_fel(pc1, pc2, args.temperature, bins)
    plot_fel_contour(X, Y, G, args.outdir / "16_FEL_PC1_PC2_contour.svg")
    plot_fel_surface(X, Y, G, args.outdir / "16_FEL_PC1_PC2_surface3D.svg")

    print("[OK] Figures generated:")
    print(f"  - {(args.outdir / '14_PCA_variance_contribution.svg').resolve()}")
    print(f"  - {(args.outdir / '14_PCA_PC1_PC2_timecolored.svg').resolve()}")
    print(f"  - {(args.outdir / '16_FEL_PC1_PC2_contour.svg').resolve()}")
    print(f"  - {(args.outdir / '16_FEL_PC1_PC2_surface3D.svg').resolve()}")


if __name__ == "__main__":
    main()
