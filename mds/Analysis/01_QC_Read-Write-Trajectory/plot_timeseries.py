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
PALETTE_9 = ["#7f7f7f"]

def load_md_data(path: Path, dt: float, ntwx: int, stride: int):
    """专门为 Amber MD 数据设计的加载函数"""
    data = np.loadtxt(path, comments='#')
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    frames = data[:, 0]
    time_ns = (frames * ntwx * stride * dt) / 1000.0
    rmsd_values = [data[:, i] for i in range(1, data.shape[1])]
    return time_ns, rmsd_values

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp", help="Input file (QC_Calpha_RMSD.dat)")
    ap.add_argument("-o", "--out", default="analysis/plots/RMSD_200ns.pdf")
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--ntwx", type=int, default=5000)
    ap.add_argument("--stride", type=int, default=10)
    args = ap.parse_args()

    # --- 新增逻辑：自动创建输出目录 ---
    output_path = Path(args.out)
    if not output_path.parent.exists():
        print(f"Creating directory: {output_path.parent}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

    set_publication_style()
    
    try:
        time_ns, rmsds = load_md_data(Path(args.inp), args.dt, args.ntwx, args.stride)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    fig, ax = plt.subplots(figsize=(5, 3.5), facecolor="white")
    ax.set_facecolor("white")

    # 绘图
    for i, val in enumerate(rmsds):
        ax.plot(time_ns, val, lw=1.0, color=PALETTE_9[i % len(PALETTE_9)], alpha=0.85)
        
        # --- 异常检测：如果 RMSD 过高，在终端提醒 ---
        max_rmsd = np.max(val)
        if max_rmsd > 5.0:
            print(f"⚠️  WARNING: High RMSD detected ({max_rmsd:.2f} Å). "
                  f"Protein structure might be unstable or misaligned.")

    ax.set_xlabel("Time (ns)", labelpad=6)
    ax.set_ylabel(r"C$\alpha$ RMSD ($\mathrm{\AA}$)", labelpad=6)
    ax.set_xlim(left=0, right=200)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    ax.tick_params(axis='y', which='both', direction='out')
    ax.tick_params(axis='y', which='major', length=6)
    ax.tick_params(axis='y', which='minor', length=3)
    # no background grid.
    ax.grid(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight', transparent=False, facecolor='white')
    print(f"Success! Final time point: {time_ns[-1]:.2f} ns")
    print(f"Plot saved to: {output_path}")

if __name__ == "__main__":
    main()
