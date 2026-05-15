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
    "#7f7f7f", "#1f77b4", "#d62728", "#2ca02c",
    "#ff7f0e", "#9467bd", "#8c564b",
    "#17becf", "#e377c2", 
]


def load_xy(path: Path):
    data = np.loadtxt(path, comments=["#", "@"])
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if data.shape[1] == 1:
        x = np.arange(1, data.shape[0] + 1)
        y = data[:, 0].astype(float)
        return x.astype(float), y
    x = data[:, 0]
    y = data[:, 1].astype(float)
    return x.astype(float), y


def frames_to_ns(frames: np.ndarray, dt: float, ntwx: int, stride: int) -> np.ndarray:
    return (frames.astype(float) * ntwx * stride * dt) / 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp", help="Input whitespace table (2+ columns recommended).")
    ap.add_argument("-o", "--out", default="plot.png")
    ap.add_argument("--title", default="")
    ap.add_argument("--xlabel", default="Time (ns)")
    ap.add_argument("--ylabel", default="Radius of Gyration (Å)")
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--ntwx", type=int, default=5000)
    ap.add_argument("--stride", type=int, default=10)
    args = ap.parse_args()

    p = Path(args.inp)
    x, y = load_xy(p)
    x = frames_to_ns(x, args.dt, args.ntwx, args.stride)

    set_publication_style()
    fig, ax = plt.subplots(figsize=(6, 4), facecolor="white")
    ax.set_facecolor("white")

    ax.plot(x, y, lw=1.3, alpha=0.9, color=PALETTE_9[1])

    if args.title:
        ax.set_title(args.title)
    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)

    x_max = float(np.nanmax(x)) if len(x) else 0.0
    ax.set_xlim(left=0, right=x_max if x_max > 0 else None)
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        y_min, y_max = 0.0, 1.0
    else:
        y_min = float(np.nanmin(y))
        y_max = float(np.nanmax(y))
        if abs(y_max - y_min) < 1e-9:
            y_max = y_min + 1.0
    span = y_max - y_min
    ax.set_ylim(y_min - 0.08 * span, y_max + 0.12 * span)
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.25))
    ax.tick_params(axis='y', which='both', direction='out')
    ax.tick_params(axis='y', which='major', length=6)
    ax.tick_params(axis='y', which='minor', length=3)

    ax.grid(False)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=600, bbox_inches='tight', transparent=False, facecolor='white')
    print(f"Plot saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
