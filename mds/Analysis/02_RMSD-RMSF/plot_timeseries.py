#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator

def set_publication_style():
    """设置顶刊学术绘图样式"""
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Cambria','Times New Roman','DejaVu Serif']
    rcParams['mathtext.fontset'] = 'stix'
    rcParams['font.size'] = 10
    rcParams['xtick.direction'] = 'in'
    rcParams['ytick.direction'] = 'in'
    rcParams['xtick.top'] = False
    rcParams['ytick.right'] = False
    rcParams['axes.linewidth'] = 1.0

# Optional publication palette for up to 9 systems (color-blind friendly, high contrast):
# 1) #1f77b4  Blue
# 2) #d62728  Red
# 3) #2ca02c  Green
# 4) #ff7f0e  Orange
# 5) #9467bd  Purple
# 6) #8c564b  Brown
# 7) #17becf  Cyan
# 8) #e377c2  Magenta
# 9) #7f7f7f  Gray
PALETTE_9 = [
    "#7f7f7f", "#1f77b4", "#d62728", "#2ca02c",
    "#ff7f0e", "#9467bd", "#8c564b",
    "#17becf", "#e377c2", 
]

def load_data(path: Path, is_rmsf: bool, dt: float, ntwx: int, stride: int):
    # 自动处理注释行
    data = np.loadtxt(path, comments=['#', '@'])
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    # 针对 RMSF/B-factor: 横坐标是残基编号，不需要换算时间
    if is_rmsf:
        x = data[:, 0]
        ylabel = "Fluctuation (Å)" if "RMSF" in path.name else "B-factor (Å²)"
        xlabel = "Residue Index"
    else:
        # 针对 RMSD: 换算为 Time (ns)
        x = (data[:, 0] * ntwx * stride * dt) / 1000.0
        ylabel = "RMSD (Å)"
        xlabel = "Time (ns)"
        
    ys = [data[:, i] for i in range(1, data.shape[1])]
    labels = [f"Col {i}" for i in range(1, data.shape[1])]
    return x, ys, labels, xlabel, ylabel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp", help="Input .dat file")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--ntwx", type=int, default=5000)
    ap.add_argument("--stride", type=int, default=10)
    args = ap.parse_args()

    # 判定是否为 RMSF 类型
    is_rmsf = any(k in args.inp.upper() for k in ["RMSF", "BFACTOR", "FLUCT"])
    
    # 自动处理输出路径
    inp_path = Path(args.inp)
    out_file = Path(args.out) if args.out else inp_path.with_suffix('.pdf')
    out_file.parent.mkdir(parents=True, exist_ok=True)

    set_publication_style()
    x, ys, labels, xlabel, ylabel = load_data(inp_path, is_rmsf, args.dt, args.ntwx, args.stride)

    fig, ax = plt.subplots(figsize=(5.5, 3.8), facecolor="white")
    ax.set_facecolor("white")

    # 绘制曲线
    for i, y in enumerate(ys):
        # RMSF 通常使用填充风格（Area plot）或者较粗的线条
        if is_rmsf:
            c = PALETTE_9[i % len(PALETTE_9)]
            ax.fill_between(x, y, alpha=0.2, color=c)
            ax.plot(x, y, lw=1.0, color=c)
        else:
            # RMSD 使用标准线图
            ax.plot(x, y, lw=1.2, alpha=0.85, color=PALETTE_9[i % len(PALETTE_9)])

    ax.set_xlabel(xlabel, labelpad=6)
    ax.set_ylabel(ylabel, labelpad=6)
    
    if not is_rmsf:
        ax.set_xlim(left=0, right=200) # 针对你的 200ns 模拟
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MultipleLocator(1.0))
        ax.yaxis.set_minor_locator(MultipleLocator(0.5))
        ax.tick_params(axis='y', which='both', direction='out')
        ax.tick_params(axis='y', which='major', length=6)
        ax.tick_params(axis='y', which='minor', length=3)
    else:
        ax.set_xlim(left=0, right=float(x.max()))
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MultipleLocator(1.0))
        ax.yaxis.set_minor_locator(MultipleLocator(0.5))
        ax.tick_params(axis='y', which='both', direction='out')
        ax.tick_params(axis='y', which='major', length=6)
        ax.tick_params(axis='y', which='minor', length=3)

    ax.grid(False)
    
    plt.tight_layout()
    plt.savefig(out_file, dpi=600, bbox_inches='tight', transparent=False, facecolor='white')
    print(f"Plot saved to: {out_file}")

if __name__ == "__main__":
    main()
