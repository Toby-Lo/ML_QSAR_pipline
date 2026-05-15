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


def load_table(path: Path) -> np.ndarray:
    rows = []
    for raw in path.read_text(errors="ignore").splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith(("#", "@")):
            continue
        # Skip gnuplot directives and separators in cpptraj .gnu outputs.
        if s.startswith(("set ", "plot ", "pause ", "unset ", "replot", "&&")):
            continue
        parts = s.replace(",", " ").split()
        nums = []
        ok = True
        for tok in parts:
            try:
                nums.append(float(tok))
            except ValueError:
                ok = False
                break
        if ok and nums:
            rows.append(nums)
    if not rows:
        raise ValueError(f"No numeric rows parsed from {path}")
    data = np.array(rows, dtype=float)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return data


def frames_to_ns(frames: np.ndarray, dt: float, ntwx: int, stride: int) -> np.ndarray:
    return (frames.astype(float) * ntwx * stride * dt) / 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp", help="Input whitespace table (2+ columns recommended).")
    ap.add_argument("-o", "--out", default="plot.png")
    ap.add_argument("--title", default="")
    ap.add_argument("--xlabel", default="Time (ns)")
    ap.add_argument("--ylabel", default="Number of Hydrogen Bonds")
    ap.add_argument(
        "--ycol",
        type=int,
        default=1,
        help="Y column index in numeric table (0-based). Default 1 means total HBond count for typical cpptraj output.",
    )
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--ntwx", type=int, default=5000)
    ap.add_argument("--stride", type=int, default=10)
    args = ap.parse_args()

    p = Path(args.inp)
    data = load_table(p)
    if data.shape[1] == 1:
        x = np.arange(1, data.shape[0] + 1, dtype=float)
        y = data[:, 0]
    else:
        if args.ycol < 0 or args.ycol >= data.shape[1]:
            raise SystemExit(f"--ycol={args.ycol} out of range for {p} with {data.shape[1]} columns")
        x = data[:, 0]
        y = data[:, args.ycol]
    x = frames_to_ns(x, args.dt, args.ntwx, args.stride)

    set_publication_style()
    fig, ax = plt.subplots(figsize=(6, 4), facecolor="white")
    ax.set_facecolor("white")

    ax.plot(x, y, lw=1.3, alpha=0.9, color=PALETTE_9[0])

    if args.title:
        ax.set_title(args.title)
    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)

    x_max = float(np.nanmax(x)) if len(x) else 0.0
    ax.set_xlim(left=0, right=x_max if x_max > 0 else None)
    y_cat = np.asarray(y, dtype=float)
    y_cat = y_cat[np.isfinite(y_cat)]
    if y_cat.size == 0:
        y_min, y_max = 0.0, 1.0
    else:
        y_min = float(np.nanmin(y_cat))
        y_max = float(np.nanmax(y_cat))
        if y_min > 0:
            y_min = 0.0
        if abs(y_max - y_min) < 1e-9:
            y_max = y_min + 1.0
    span = y_max - y_min
    ax.set_ylim(0.0, y_max + 0.12 * span)
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
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
