#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

def set_publication_style():
    """设置顶刊级别的绘图参数"""
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman', 'Cambria', 'DejaVu Serif']
    rcParams['mathtext.fontset'] = 'stix'
    rcParams['font.size'] = 10
    rcParams['axes.titlesize'] = 12
    rcParams['axes.labelsize'] = 11
    # 热图通常保持四边框，刻度向内更精致
    rcParams['xtick.direction'] = 'in'
    rcParams['ytick.direction'] = 'in'
    rcParams['axes.linewidth'] = 1.0

def load_matrix(path: Path):
    txt = path.read_text(errors="ignore").strip().splitlines()
    txt = [l for l in txt if l and not l.startswith(("#","@"))]
    arr = np.loadtxt(txt)
    if arr.ndim == 2 and arr.shape[1] == 3:
        x = np.unique(arr[:,0])
        y = np.unique(arr[:,1])
        z = arr[:,2].reshape(len(y), len(x))
        return z
    if arr.ndim == 2:
        return arr
    raise ValueError("Unrecognized format")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp", help="Input matrix file")
    ap.add_argument("-o", "--out", default="heatmap_200ns.pdf")
    ap.add_argument("--title", default="")
    # 物理时间相关参数
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--ntwx", type=int, default=5000)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--zlabel", default="Correlation / Distance (Å)")
    args = ap.parse_args()

    set_publication_style()
    z = load_matrix(Path(args.inp))
    
    # 计算时间轴范围 (假设矩阵的行/列对应的是提取后的帧)
    num_frames = z.shape[0]
    total_time_ns = (num_frames * args.ntwx * args.stride * args.dt) / 1000.0

    fig, ax = plt.subplots(figsize=(5, 4.2)) # 稍宽一点给颜色条留位置
    
    # 绘制热图
    # extent 参数将坐标轴从“像素索引”转变为“物理时间”
    im = ax.imshow(z, origin="lower", aspect="equal", 
                   cmap="RdBu_r", # 常用红白蓝，适合表现正负相关性
                   extent=[0, total_time_ns, 0, total_time_ns],
                   interpolation="gaussian") # 高斯插值使图片看起来更丝滑
    
    # 颜色条精修
    cbar = fig.colorbar(im, ax=ax, pad=0.03, aspect=20)
    cbar.set_label(args.zlabel)
    cbar.outline.set_linewidth(1.0)

    # 坐标轴美化
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Time (ns)")
    
    if args.title:
        ax.set_title(args.title, fontweight='bold', pad=12)

    plt.tight_layout()
    
    output_path = Path(args.out)
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"Heatmap saved! Matrix dimension: {z.shape}, Max Time: {total_time_ns:.2f} ns")

if __name__ == "__main__":
    main()