'''
python3 ../../Analysis/zn_distances/plot_zn_distances.py analysis/ZN221_CYM161_ZN_SG.dat -o analysis/plots/ZN221_CYM161.png

batch running 
python3 ../../Analysis/zn_distances/plot_zn_distances.py
'''
#!/usr/bin/env python3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib import rcParams

def set_publication_style():
    """设置学术顶刊级别的绘图参数"""
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Cambria','Times New Roman', 'DejaVu Serif']
    rcParams['mathtext.fontset'] = 'stix'
    rcParams['font.size'] = 10
    rcParams['axes.linewidth'] = 0.8
    rcParams['xtick.direction'] = 'in'
    rcParams['ytick.direction'] = 'in'
    rcParams['xtick.top'] = True
    rcParams['ytick.right'] = True

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def plot_zn_group(ax, data_paths, title, color_map, dt=0.002, ntwx=5000, stride=1):
    """在指定子图上绘制一组 Zn 的距离曲线"""
    all_dist = []
    for path in data_paths:
        if not path.exists(): continue
        
        # 加载数据 (假设 cpptraj 输出两列：Frame 和 Distance)
        data = np.loadtxt(path, comments=['#', '@'])
        time = (data[:, 0] * ntwx * stride * dt) / 1000.0  # 转换为 ns
        dist = data[:, 1]
        finite_dist = dist[np.isfinite(dist)]
        if finite_dist.size:
            all_dist.append(finite_dist)
        
        label = path.stem.replace("ZN", "Zn").replace("_ZN_SG", "")
        color = color_map.get(path.name, '#7f8c8d')
        
        # 绘制原始数据 (极浅颜色)
        ax.plot(time, dist, lw=0.3, color=color, alpha=0.2)
        
        # 绘制移动平均线 (平滑)
        window = max(1, len(dist) // 50)
        if window > 1:
            dist_smooth = moving_average(dist, window)
            time_smooth = time[window-1:]
            ax.plot(time_smooth, dist_smooth, lw=1.2, color=color, label=label)
        else:
            ax.plot(time, dist, lw=1.0, color=color, label=label)

    ax.set_title(title, loc='left', fontsize=11, fontweight='bold')
    ax.set_ylabel(r"Distance ($\mathrm{\AA}$)")
    if all_dist:
        y = np.concatenate(all_dist)
        y_min = float(np.min(y))
        y_max = float(np.max(y))
        span = max(y_max - y_min, 0.15)
        # 动态范围：底部小留白，顶部多留一点给图例，避免遮挡主要曲线
        low = y_min - 0.08 * span
        high = y_max + 0.22 * span
        ax.set_ylim(low, high)
    ax.set_xlim(0, 200)
    ax.margins(x=0)
    ax.grid(False)
    ax.legend(
        loc='upper right',
        fontsize=7,
        ncol=2,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor='black',
    )

def main():
    set_publication_style()
    data_dir = Path("analysis")
    out_file = Path("analysis/plots/00_Zn_Coordination_Stability.svg")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # 定义颜色方案
    colors = {
        # Zn221 Group
        "ZN221_CYM161_ZN_SG.dat": "#2c3e50", "ZN221_CYM208_ZN_SG.dat": "#e74c3c",
        "ZN221_CYM210_ZN_SG.dat": "#27ae60", "ZN221_CYM215_ZN_SG.dat": "#f39c12",
        # Zn222 Group
        "ZN222_CYM33_ZN_SG.dat": "#2980b9", "ZN222_CYM35_ZN_SG.dat": "#8e44ad",
        "ZN222_CYM43_ZN_SG.dat": "#c0392b", "ZN222_CYM49_ZN_SG.dat": "#16a085",
        # Zn223 Group
        "ZN223_CYM43_ZN_SG.dat": "#d35400", "ZN223_CYM58_ZN_SG.dat": "#7f8c8d",
        "ZN223_CYM63_ZN_SG.dat": "#2c3e50", "ZN223_CYM69_ZN_SG.dat": "#bdc3c7"
    }

    # 分组
    group1 = [data_dir / f"ZN221_CYM{r}_ZN_SG.dat" for r in [161, 208, 210, 215]]
    group2 = [data_dir / f"ZN222_CYM{r}_ZN_SG.dat" for r in [33, 35, 43, 49]]
    group3 = [data_dir / f"ZN223_CYM{r}_ZN_SG.dat" for r in [43, 58, 63, 69]]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 7), sharex=True)

    plot_zn_group(ax1, group1, "A: Zn221 Coordination (Single Site)", colors)
    plot_zn_group(ax2, group2 + group3, "B: Zn222 & Zn223 Cluster (Shared Site)", colors)

    ax2.set_xlabel("Time (ns)")
    plt.tight_layout()
    plt.savefig(out_file, dpi=600, bbox_inches='tight')
    print(f"Publication plot saved to: {out_file}")

if __name__ == "__main__":
    main()
