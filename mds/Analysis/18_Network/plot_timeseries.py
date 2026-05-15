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
    rcParams['font.serif'] = ['Cambria', 'Times New Roman', 'DejaVu Serif']
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
    "#1f77b4", "#d62728", "#2ca02c",
    "#ff7f0e", "#9467bd", "#8c564b",
    "#17becf", "#e377c2", "#7f7f7f",
]


def load_xy(path: Path):
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if data.shape[1] == 1:
        x = np.arange(1, data.shape[0] + 1)
        y = data[:, 0]
        return x, [y], ["y"]
    x = data[:, 0]
    ys = [data[:, i] for i in range(1, data.shape[1])]
    labels = [f"col{i}" for i in range(1, data.shape[1])]
    return x, ys, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp", help="Input whitespace table (2+ columns recommended).")
    ap.add_argument("-o", "--out", default="plot.png")
    ap.add_argument("--title", default="")
    ap.add_argument("--xlabel", default="Frame")
    ap.add_argument("--ylabel", default="Value")
    args = ap.parse_args()

    p = Path(args.inp)
    x, ys, labels = load_xy(p)

    set_publication_style()
    fig, ax = plt.subplots(figsize=(6, 4), facecolor="white")
    ax.set_facecolor("white")

    for i, (y, lab) in enumerate(zip(ys, labels)):
        ax.plot(x, y, lw=1.2, alpha=0.85, color=PALETTE_9[i % len(PALETTE_9)], label=lab if len(ys) > 1 else None)

    if args.title:
        ax.set_title(args.title)
    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)

    x_max = float(np.nanmax(x)) if len(x) else 0.0
    ax.set_xlim(left=0, right=x_max if x_max > 0 else None)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    ax.tick_params(axis='y', which='both', direction='out')
    ax.tick_params(axis='y', which='major', length=6)
    ax.tick_params(axis='y', which='minor', length=3)

    ax.grid(False)
    if len(ys) > 1:
        ax.legend(frameon=True, fancybox=False, framealpha=1.0, edgecolor='black', fontsize=8)

    plt.tight_layout()
    plt.savefig(args.out, dpi=600, bbox_inches='tight', transparent=False, facecolor='white')


if __name__ == "__main__":
    main()
