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

Batch example (run in root path)
# rm -f mds/runs/*/analysis/plots/16_*

for d in mds/runs/*; do \
  [ -d "$d" ] || continue; \
  echo "Drawing for $d..."; \
  python3 mds/Analysis/16_FEL/pca_fel_publication_plots.py \
    --projection "$d/analysis/PCA_projection.dat" \
    --outdir "$d/analysis/plots"; \
done

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
from matplotlib.colors import LinearSegmentedColormap
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


def get_morandi_cmap() -> LinearSegmentedColormap:
    # 莫兰迪科学色系：深紫 (阱底) -> 灰青 -> 莫兰迪粉 -> 泥棕 (势垒)
    colors = ["#4A3F5E", "#6F8A91", "#C7A8A6", "#9C856E"]
    return LinearSegmentedColormap.from_list("morandi", colors)


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


def plot_variance(ratio: np.ndarray, out: Path, top_n: int, title: str = "PCA Variance Contribution") -> None:
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

    ax1.set_title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=600, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)


def plot_pc12_scatter(pc1: np.ndarray, pc2: np.ndarray, t_ns: np.ndarray, out: Path, ratio: np.ndarray, title: str = "Trajectory Projection on PC1-PC2") -> None:
    fig, ax = plt.subplots(figsize=(6.3, 5.2), facecolor="white")
    ax.set_facecolor("white")

    sc = ax.scatter(pc1, pc2, c=t_ns, cmap="turbo", s=10, alpha=0.88, linewidths=0)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Time (ns)")

    ax.set_xlabel(f"PC1 ({ratio[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ratio[1]*100:.1f}%)")
    ax.set_title(title)
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


def gaussian_smooth2d(arr: np.ndarray, sigma: float = 2.5) -> np.ndarray:
    if sigma <= 0:
        return arr
    radius = max(1, int(np.ceil(3.0 * sigma)))
    k = gaussian_kernel1d(sigma, radius)

    # Convolve along x then y (separable Gaussian).
    tmp = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), axis=1, arr=arr)
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), axis=0, arr=tmp)
    return out


def plot_fel_contour(X: np.ndarray, Y: np.ndarray, G: np.ndarray, out: Path, ratio: np.ndarray, title: str = "Free Energy Landscape (PC1-PC2)") -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.3), facecolor="white")
    ax.set_facecolor("white")

    vmax = np.nanpercentile(G, 97)
    vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
    levels = np.linspace(0, vmax, 24)
    cmap = get_morandi_cmap()
    cf = ax.contourf(X, Y, G, levels=levels, cmap=cmap)
    ax.contour(X, Y, G, levels=levels[::3], colors="white", linewidths=0.35, alpha=0.55)

    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(r"$\Delta G$ (kcal/mol)")

    ax.set_xlabel(f"PC1 ({ratio[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ratio[1]*100:.1f}%)")
    ax.set_title(title)
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(out, dpi=600, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)


def plot_fel_surface(X: np.ndarray, Y: np.ndarray, G: np.ndarray, out: Path, ratio: np.ndarray, title: str = "Free Energy Landscape (3D Surface)") -> None:
    fig = plt.figure(figsize=(7.2, 5.8), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")

    # 去掉背景的灰色网格
    ax.grid(False)

    # 设置透明面板，并为后侧和底部面板添加黑色边框（保留棱线，避免前面的棱遮挡）
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # 面板背景透明
        axis.pane.set_edgecolor('black')           # 边框（棱）设为黑色
        axis.pane.set_linewidth(1.0)               # 棱的线宽
        axis.pane.set_alpha(1.0)                   # 确保边框不透明
        axis.line.set_color('black')               # 坐标轴的基线也统一为黑色

    # Clip extreme NaN/inf regions for cleaner surface
    Gp = np.array(G, copy=True)
    vmax = np.nanpercentile(Gp, 95)
    Gp = np.where(np.isfinite(Gp), np.minimum(Gp, vmax), np.nan)

    cmap = get_morandi_cmap()
    surf = ax.plot_surface(
        X, Y, Gp,
        cmap=cmap,
        linewidth=0,
        antialiased=True,
        alpha=0.95,
        rstride=1,
        cstride=1,
        shade=True
    )

    zmin = float(np.nanmin(Gp)) if np.any(np.isfinite(Gp)) else 0.0
    zmax = float(np.nanmax(Gp)) if np.any(np.isfinite(Gp)) else 1.0
    # 增加等高线层数（从12层提高到18层），可以让底部的色彩渐变极其细腻顺滑
    levels = np.linspace(zmin, float(vmax), 18)
    offset_z = zmin - 0.08  # 稍微往下压一点点，避免与3D曲面的最低能量点重叠粘连

    # 【核心改进】将底部的线状投影 升级为 文献风的“实体色彩填充面”
    # 1. 先绘制“实体色彩填充面” (Contourf) —— 这是向文献靠拢的关键！
    # alpha=0.65 既保证了底部色彩足够饱满斑斓，又不会过于抢戏而遮挡3D坐标网格的透视感
    ax.contourf(
        X, Y, Gp,
        zdir='z',
        offset=offset_z,
        cmap=cmap,
        levels=levels,
        alpha=0.65
    )

    # 2. 再在其上叠加一层“极细的白色或淡色边界线” (Contour)
    # 这能勾勒出像文献中那样精致、清晰的势阱分界边缘
    ax.contour(
        X, Y, Gp, zdir='z', offset=offset_z, colors='white',
        levels=levels[::2], linewidths=0.5, alpha=0.6
    )
    ax.set_zlim(offset_z, zmax)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.65, pad=0.02)
    cbar.set_label(r"$\Delta G$ (kcal/mol)")

    ax.set_xlabel(f"PC1 ({ratio[0]*100:.1f}%)", labelpad=8)
    ax.set_ylabel(f"PC2 ({ratio[1]*100:.1f}%)", labelpad=8)
    
    # 强制将 Z 轴的带刻度支柱固定在最左侧 (利用 mplot3d 底层 _axinfo 属性)
    ax.zaxis._axinfo['juggled'] = (1, 2, 0)
    
    # 隐藏 3D 坐标系自带的 Z 轴标签，防止随视角乱跑
    ax.set_zlabel("")
    # 使用 2D 绝对坐标强行将 Z 轴标签固定在整个图表的最左侧
    ax.text2D(-0.09, 0.5, r"$\Delta G$ (kcal/mol)", transform=ax.transAxes, rotation=90, va="center", ha="center")
    
    ax.set_title(title)

    # 调整 3D 视角：elev 是仰角（上下看），azim 是方位角（左右转）
    ax.view_init(elev=20, azim=-50)

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
    ap.add_argument("--system", type=str, default="", help="System name to prepend to plot titles")
    args = ap.parse_args()

    set_publication_style()
    args.outdir.mkdir(parents=True, exist_ok=True)

    if not args.projection.exists():
        raise SystemExit(f"[ERROR] Projection file not found: {args.projection}")
        
    # 自动从 outdir 路径中推断体系名称（如果手动输入了 system 则使用手动输入）
    system_name = args.system.strip()
    if not system_name:
        parts = args.outdir.resolve().parts
        if "runs" in parts:
            idx = parts.index("runs")
            if idx + 1 < len(parts):
                system_name = parts[idx + 1]
    
    prefix = f"{system_name.upper()} - " if system_name else ""

    frames, pcs = load_projection(args.projection)
    t_ns = frames_to_ns(frames, args.dt, args.ntwx, args.stride)

    ratio = explained_variance_ratio(pcs)
    plot_variance(ratio, args.outdir / "14_PCA_variance_contribution.svg", args.top_pc, title=f"{prefix}PCA Variance Contribution")

    pc1 = pcs[:, 0]
    pc2 = pcs[:, 1]
    plot_pc12_scatter(pc1, pc2, t_ns, args.outdir / "14_PCA_PC1_PC2_timecolored.svg", ratio, title=f"{prefix}Trajectory Projection on PC1-PC2")

    bins = args.bins
    if bins <= 0:
        # Auto bins: enough detail for dense trajectories, but avoid overly sparse maps.
        n = len(pc1)
        bins = int(np.clip(np.sqrt(n) * 1.7, 60, 140))
    X, Y, G = compute_fel(pc1, pc2, args.temperature, bins)
    plot_fel_contour(X, Y, G, args.outdir / "16_FEL_PC1_PC2_contour.svg", ratio, title=f"{prefix}Free Energy Landscape (PC1-PC2)")
    plot_fel_surface(X, Y, G, args.outdir / "16_FEL_PC1_PC2_surface3D.svg", ratio, title=f"{prefix}Free Energy Landscape (3D Surface)")

    print("[OK] Figures generated:")
    print(f"  - {(args.outdir / '14_PCA_variance_contribution.svg').resolve()}")
    print(f"  - {(args.outdir / '14_PCA_PC1_PC2_timecolored.svg').resolve()}")
    print(f"  - {(args.outdir / '16_FEL_PC1_PC2_contour.svg').resolve()}")
    print(f"  - {(args.outdir / '16_FEL_PC1_PC2_surface3D.svg').resolve()}")


if __name__ == "__main__":
    main()
