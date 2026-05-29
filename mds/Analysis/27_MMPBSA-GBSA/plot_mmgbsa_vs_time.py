#!/usr/bin/env python3
# python3 ../../Analysis/27_MMPBSA-GBSA/plot_mmgbsa_vs_time.py MMGBSA_vs_time.dat -o analysis/plots/27_MMGBSA_vs_time.png --window 25
import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator


def set_publication_style() -> None:
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = ["Cambria", "Times New Roman", "DejaVu Serif"]
    rcParams["mathtext.fontset"] = "stix"
    rcParams["font.size"] = 10
    rcParams["xtick.direction"] = "in"
    rcParams["ytick.direction"] = "in"
    rcParams["xtick.top"] = False
    rcParams["ytick.right"] = False
    rcParams["axes.linewidth"] = 1.0


def rolling_mean(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y
    if y.size == 0:
        return y
    if window > y.size:
        window = y.size
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(y, kernel, mode="valid")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot MM/GBSA vs frame index from MMGBSA_vs_time.dat (from GBSA-vs-Time.i)."
    )
    ap.add_argument("inp", nargs="?", default="MMGBSA_vs_time.dat", help="Input table (2 cols: frame dG).")
    ap.add_argument(
        "-o",
        "--out",
        default="MMGBSA_vs_time.pdf",
        help="Output figure path (pdf/png).",
    )
    ap.add_argument("--title", default="", help="Optional plot title.")
    ap.add_argument(
        "--xmode",
        choices=["frame", "ns"],
        default="frame",
        help="X-axis mode: frame index or converted time (ns).",
    )
    ap.add_argument(
        "--ns-per-frame",
        type=float,
        default=2.0,
        help="Only used when --xmode ns. Time step represented by one MMGBSA point (ns/frame).",
    )
    ap.add_argument("--xmax", type=float, default=None, help="Optional fixed x-axis max (e.g., 100 for 100 frames).")
    ap.add_argument("--xlabel", default=None, help="X-axis label. Default follows --xmode.")
    ap.add_argument("--ylabel", default=r"$\Delta G_{\mathrm{GBSA}}$ (kcal/mol)", help="Y-axis label.")
    ap.add_argument(
        "--window",
        type=int,
        default=0,
        help="Rolling mean window (in points). 0 disables. Suggest 10-50 for smoothing.",
    )
    ap.add_argument("--dpi", type=int, default=450)
    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not inp.exists():
        raise SystemExit(f"Input file not found: {inp}")
    if inp.stat().st_size == 0:
        raise SystemExit(
            f"Input file is empty: {inp}. Re-generate it first, e.g.:\n"
            f"  bash ../../Analysis/27_MMPBSA-GBSA/GBSA-vs-Time.i"
        )

    data = np.loadtxt(inp)
    if data.size == 0:
        raise SystemExit(
            f"No numeric data parsed from: {inp}. Check file format or regenerate it with GBSA-vs-Time.i."
        )
    if data.ndim == 1:
        data = data.reshape(-1, 2)
    if data.shape[1] < 2:
        raise SystemExit(f"Expected >=2 columns in {inp}, got shape={data.shape}")

    x = data[:, 0]
    y = data[:, 1]
    if args.xmode == "ns":
        x_plot = (x - 1.0) * float(args.ns_per_frame)
        default_xlabel = "Time (ns)"
    else:
        x_plot = x
        default_xlabel = "Frame Index"
    xlabel = args.xlabel if args.xlabel is not None else default_xlabel

    set_publication_style()
    fig, ax = plt.subplots(figsize=(6.0, 3.6), facecolor="white")
    ax.set_facecolor("white")

    ax.plot(x_plot, y, lw=0.9, color="#1f77b4", alpha=0.55, label="Per-frame")

    if args.window and args.window > 1:
        y_sm = rolling_mean(y, args.window)
        if y_sm.size == 0:
            raise SystemExit(f"Not enough points ({y.size}) for rolling window {args.window}.")
        x_sm = x_plot[args.window - 1 :]
        ax.plot(x_sm, y_sm, lw=1.6, color="#d62728", alpha=0.95, label=f"Rolling mean (w={args.window})")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(args.ylabel)
    if args.title:
        ax.set_title(args.title)
    x_min = float(np.nanmin(x_plot))
    x_max = float(np.nanmax(x_plot))
    x_right = float(args.xmax) if args.xmax is not None else x_max
    ax.set_xlim(x_min, x_right)
    y_all = [y]
    if args.window and args.window > 1:
        y_all.append(y_sm)
    y_cat = np.concatenate(y_all) if y_all else y
    y_min = float(np.nanmin(y_cat))
    y_max = float(np.nanmax(y_cat))
    y_span = max(y_max - y_min, 1e-6)
    y_pad = 0.12 * y_span
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.yaxis.set_major_locator(MultipleLocator(5.0))
    ax.yaxis.set_minor_locator(MultipleLocator(1.0))
    ax.tick_params(axis="y", which="both", direction="out")
    ax.tick_params(axis="y", which="major", length=6)
    ax.tick_params(axis="y", which="minor", length=3)
    ax.grid(False)
    ax.legend(frameon=True, fancybox=False, framealpha=1.0, edgecolor="black", fontsize=8)

    plt.tight_layout()
    plt.savefig(out, dpi=args.dpi, bbox_inches="tight", transparent=False, facecolor="white")
    print(f"Plot saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
