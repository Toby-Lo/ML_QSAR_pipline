#!/usr/bin/env python3
"""
Build occupancy table + Top-N occupancy plot from cpptraj HBond avgout files.

Run from a simulation directory, e.g.:
  python3 ../../Analysis/09_HBond-Complex/hbond_occupancy_report.py \
    --p2l analysis/HBond_PL_p2l.avg.dat \
    --l2p analysis/HBond_PL_l2p.avg.dat \
    --out-csv analysis/HBond_PL_occupancy_summary.csv \
    --out-fig analysis/plots/09_HBond_PL_occupancy_top15.svg \
    --topn 15
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams


def set_publication_style() -> None:
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Cambria', 'Times New Roman', 'DejaVu Serif']
    rcParams['mathtext.fontset'] = 'stix'
    rcParams['font.size'] = 10
    rcParams['xtick.direction'] = 'in'
    rcParams['ytick.direction'] = 'in'
    rcParams['xtick.top'] = False
    rcParams['ytick.right'] = False
    rcParams['axes.linewidth'] = 1.0


PALETTE_9 = [
    "#7f7f7f", "#1f77b4", "#d62728", "#2ca02c",
    "#ff7f0e", "#9467bd", "#8c564b", "#17becf", "#e377c2",
]


def _try_parse_numeric_tail(tokens: List[str]) -> Tuple[float, float, float] | None:
    vals: List[float] = []
    for tok in reversed(tokens):
        t = tok.replace('%', '')
        try:
            vals.append(float(t))
            if len(vals) == 3:
                break
        except ValueError:
            continue
    if len(vals) < 3:
        return None
    ang, dist, occ = vals[0], vals[1], vals[2]
    return occ, dist, ang


def parse_avg_file(path: Path, direction: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    rows = []
    for raw in path.read_text(errors="ignore").splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith(("#", "@")):
            continue

        tokens = s.split()
        if len(tokens) < 3:
            continue

        donor = tokens[0]
        acceptor = tokens[1] if len(tokens) > 1 else ""
        tail = _try_parse_numeric_tail(tokens)
        if tail is None:
            continue
        occ, dist, ang = tail

        rows.append(
            {
                "Direction": direction,
                "Donor": donor,
                "Acceptor": acceptor,
                "Occupancy (%)": occ,
                "Avg Distance (A)": dist,
                "Avg Angle (deg)": ang,
                "Pair": f"{donor} -> {acceptor}",
            }
        )

    if not rows:
        raise ValueError(f"No parseable HBond rows in: {path}")

    df = pd.DataFrame(rows)
    df = df.sort_values("Occupancy (%)", ascending=False).reset_index(drop=True)
    return df


def plot_topn(df: pd.DataFrame, out_fig: Path, topn: int) -> None:
    use = df.sort_values("Occupancy (%)", ascending=False).head(max(1, topn)).copy()
    use = use.iloc[::-1]

    set_publication_style()
    fig_h = max(4.0, 0.33 * len(use) + 1.6)
    fig, ax = plt.subplots(figsize=(8.2, fig_h), facecolor="white")
    ax.set_facecolor("white")

    ax.barh(
        np.arange(len(use)),
        use["Occupancy (%)"].to_numpy(dtype=float),
        color=PALETTE_9[1],
        alpha=0.9,
    )
    ax.set_yticks(np.arange(len(use)))
    ax.set_yticklabels(use["Pair"].tolist())
    ax.set_xlabel("Occupancy (%)")
    ax.set_ylabel("Hydrogen Bond")
    ax.set_xlim(left=0)
    ax.grid(False)

    out_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_fig, dpi=600, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Create HBond occupancy summary CSV and TopN plot.")
    ap.add_argument("--p2l", type=Path, default=Path("analysis/HBond_PL_p2l.avg.dat"))
    ap.add_argument("--l2p", type=Path, default=Path("analysis/HBond_PL_l2p.avg.dat"))
    ap.add_argument("--out-csv", type=Path, default=Path("analysis/HBond_PL_occupancy_summary.csv"))
    ap.add_argument("--out-fig", type=Path, default=Path("analysis/plots/HBond_PL_occupancy_top15.svg"))
    ap.add_argument("--topn", type=int, default=15)
    args = ap.parse_args()

    df_p2l = parse_avg_file(args.p2l, "Protein -> Ligand")
    df_l2p = parse_avg_file(args.l2p, "Ligand -> Protein")
    df = pd.concat([df_p2l, df_l2p], ignore_index=True)
    df = df.sort_values("Occupancy (%)", ascending=False).reset_index(drop=True)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    plot_topn(df, args.out_fig, args.topn)

    print(f"[OK] CSV saved: {args.out_csv.resolve()}")
    print(f"[OK] Figure saved: {args.out_fig.resolve()}")


if __name__ == "__main__":
    main()
