#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_matrix(path: Path):
    # supports:
    # (a) plain whitespace matrix
    # (b) 3-column x y val grids (gnuplot-like)
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
    ap.add_argument("inp")
    ap.add_argument("-o", "--out", default="heatmap.png")
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    z = load_matrix(Path(args.inp))
    plt.figure(figsize=(6,5))
    plt.imshow(z, origin="lower", aspect="auto", cmap="viridis")
    plt.colorbar(label="Value")
    if args.title:
        plt.title(args.title)
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)

if __name__ == "__main__":
    main()
