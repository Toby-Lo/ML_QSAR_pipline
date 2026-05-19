#!/usr/bin/env python3
"""
Aggregate MD analysis across systems under mds/runs/*/analysis.

Outputs (SVG):
  - 02_Calpha_RMSD_multi.svg
  - 03_Ligand_RMSD_multi.svg
  - 06_RoG_multi.svg
  - 07_SASA_protein_multi.svg
  - 07_SASA_ligand_multi.svg
  - 09_HBond_total_multi.svg

Outputs (CSV):
  - md_summary_all_systems.csv
  - md_timeseries_long.csv

python3 mds/aggregate_md_results.py check --runs-root mds/runs

python3 mds/aggregate_md_results.py --reference a1a0m --runs-root mds/runs --outdir mds/aggregate_outputs --plot-zn-violin
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Circle, FancyArrowPatch


def read_table(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    try:
        arr = np.loadtxt(path, comments=["#", "@"])
    except Exception:
        return None
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def pick_first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def to_time_ns(frames: np.ndarray, dt: float, ntwx: int, stride: int) -> np.ndarray:
    return (frames.astype(float) * ntwx * stride * dt) / 1000.0



def extract_xy(path: Path, dt: float, ntwx: int, stride: int, ycol: int = 1) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    arr = read_table(path)
    if arr is None or arr.shape[1] <= ycol:
        return None
    x = to_time_ns(arr[:, 0], dt=dt, ntwx=ntwx, stride=stride)
    y = arr[:, ycol].astype(float)
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return None
    return x[mask], y[mask]


def summarize(y: np.ndarray) -> Dict[str, float]:
    return {
        "n": float(y.size),
        "mean": float(np.mean(y)),
        "std": float(np.std(y, ddof=1)) if y.size > 1 else 0.0,
        "min": float(np.min(y)),
        "max": float(np.max(y)),
        "median": float(np.median(y)),
    }


def summarize_last_n(y: np.ndarray, n_last: int) -> Dict[str, float]:
    if y.size == 0:
        return {"n": 0.0, "mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan, "median": np.nan}
    n_use = min(int(n_last), int(y.size))
    ys = y[-n_use:]
    return summarize(ys)


def read_mmgbsa_last50_summary(path: Path) -> Optional[Dict[str, float]]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None
    r = df.iloc[0]
    return {
        "mmgbsa_last50_n_points": float(pd.to_numeric(r.get("n_points"), errors="coerce")),
        "mmgbsa_last50_mean": float(pd.to_numeric(r.get("mean_kcal_per_mol"), errors="coerce")),
        "mmgbsa_last50_sd": float(pd.to_numeric(r.get("sd_kcal_per_mol"), errors="coerce")),
        "mmgbsa_last50_sem": float(pd.to_numeric(r.get("sem_kcal_per_mol"), errors="coerce")),
    }


def read_hbond_top1(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None

    def pick_col(*candidates: str) -> Optional[str]:
        cols = {c.strip().lower(): c for c in df.columns}
        for cand in candidates:
            key = cand.strip().lower()
            if key in cols:
                return cols[key]
        return None

    # Support both "normalized" column names and the raw report output from
    # Analysis/09_HBond-Complex/hbond_occupancy_report.py.
    donor_col = pick_col("donor", "Donor")
    acceptor_col = pick_col("acceptor", "Acceptor")
    occ_col = pick_col("occupancy_pct", "occupancy (%)", "occupancy", "Occupancy (%)", "Occupancy")
    dist_col = pick_col("avg_distance", "avg distance (a)", "avg distance (å)", "Avg Distance (A)", "Avg Distance (Å)")
    angle_col = pick_col("avg_angle", "avg angle (deg)", "Avg Angle (deg)")

    # Use highest occupancy row when possible.
    if occ_col is not None:
        dfx = df.sort_values(occ_col, ascending=False).reset_index(drop=True)
    else:
        dfx = df.reset_index(drop=True)
    r = dfx.iloc[0]

    donor_val = str(r.get(donor_col, "")) if donor_col is not None else ""
    acceptor_val = str(r.get(acceptor_col, "")) if acceptor_col is not None else ""
    occ_val = float(pd.to_numeric(r.get(occ_col), errors="coerce")) if occ_col is not None else np.nan
    dist_val = float(pd.to_numeric(r.get(dist_col), errors="coerce")) if dist_col is not None else np.nan
    angle_val = float(pd.to_numeric(r.get(angle_col), errors="coerce")) if angle_col is not None else np.nan

    return {
        "hbond_top1_donor": donor_val,
        "hbond_top1_acceptor": acceptor_val,
        "hbond_top1_occupancy_pct": occ_val,
        "hbond_top1_avg_distance": dist_val,
        "hbond_top1_avg_angle": angle_val,
    }


def extract_zn_last_50ns_stats(path: Path, dt: float, ntwx: int, stride: int = 1) -> Optional[Tuple[float, float]]:
    arr = read_table(path)
    if arr is None or arr.shape[1] < 2:
        return None
    t = (arr[:, 0].astype(float) * ntwx * stride * dt) / 1000.0
    y = arr[:, 1].astype(float)
    mask = np.isfinite(t) & np.isfinite(y)
    t = t[mask]
    y = y[mask]
    if len(t) == 0:
        return None
    t_max = np.max(t)
    last_mask = t >= max(0.0, t_max - 50.0)
    y_last = y[last_mask]
    if len(y_last) == 0:
        return None
    return float(np.mean(y_last)), float(np.std(y_last, ddof=1) if len(y_last) > 1 else 0.0)


def extract_zn_last_50ns_raw(path: Path, dt: float, ntwx: int, stride: int = 1) -> Optional[np.ndarray]:
    arr = read_table(path)
    if arr is None or arr.shape[1] < 2:
        return None
    t = (arr[:, 0].astype(float) * ntwx * stride * dt) / 1000.0
    y = arr[:, 1].astype(float)
    mask = np.isfinite(t) & np.isfinite(y)
    t = t[mask]
    y = y[mask]
    if len(t) == 0:
        return None
    t_max = np.max(t)
    last_mask = t >= max(0.0, t_max - 50.0)
    y_last = y[last_mask]
    if len(y_last) == 0:
        return None
    return y_last


def parse_final_gbsa_delta_g(path: Path) -> Optional[Dict[str, float]]:
    """
    Parse DELTA G and energy components from FINAL_GBSA.dat generated by MMPBSA.py.
    It specifically looks for the "Differences (Complex - Receptor - Ligand):"
    section to extract final binding energy contributions.
    """
    if not path.exists():
        return None
    txt = path.read_text(encoding="utf-8", errors="ignore")

    results: Dict[str, float] = {}

    # Isolate the "Differences" section text for accurate parsing.
    diff_match = re.search(
        r"Differences \(Complex - Receptor - Ligand\):", txt, re.IGNORECASE
    )

    if diff_match:
        parse_text = txt[diff_match.end():]
        val_re = r"([+-]?(?:\d+\.?\d*|\.\d+)(?:[Ee][+-]?\d+)?)"

        def parse_one_component(name_in_file: str, text: str) -> Optional[Dict[str, float]]:
            # Format 1: Table row, e.g., "VDWAALS   -53.05   4.56   0.20"
            table_pat = re.compile(
                r"^\s*" + re.escape(name_in_file) + r"\s+" + val_re + r"\s+" + val_re + r"\s+" + val_re,
                re.IGNORECASE | re.MULTILINE,
            )
            m_table = table_pat.search(text)
            if m_table:
                return {
                    "mean": float(m_table.group(1)),
                    "std": float(m_table.group(2)),
                    "sem": float(m_table.group(3)),
                }

            # Format 2: Key-value pair, e.g., "VDWAALS = -53.05 +/- 4.56"
            kv_pat = re.compile(
                r"^\s*" + re.escape(name_in_file) + r"\s*=\s*" + val_re + r"\s*\+/-\s*" + val_re,
                re.IGNORECASE | re.MULTILINE,
            )
            m_kv = kv_pat.search(text)
            if m_kv:
                return {
                    "mean": float(m_kv.group(1)),
                    "std": float(m_kv.group(2)),
                }
            return None

        # Components to extract
        components = {
            "VDWAALS": "vdwaals",
            "EEL": "eel",
            "EGB": "egb",
            "ESURF": "esurf",
        }
        for name_in_file, out_prefix in components.items():
            vals = parse_one_component(name_in_file, parse_text)
            if vals:
                results[f"final_gbsa_{out_prefix}_mean"] = vals["mean"]
                results[f"final_gbsa_{out_prefix}_std"] = vals["std"]
                if "sem" in vals:
                    results[f"final_gbsa_{out_prefix}_sem"] = vals["sem"]

        # Handle DELTA G (which has multiple possible names)
        delta_g_vals = parse_one_component("DELTA TOTAL", parse_text)
        if not delta_g_vals:
            delta_g_vals = parse_one_component("DELTA G binding", parse_text)

        if delta_g_vals:
            results["final_gbsa_delta_g_mean"] = delta_g_vals["mean"]
            results["final_gbsa_delta_g_std"] = delta_g_vals["std"]
            if "sem" in delta_g_vals:
                results["final_gbsa_delta_g_sem"] = delta_g_vals["sem"]

    # Fallback for DELTA G if it wasn't found in the "Differences" section
    # This maintains compatibility with older formats that might not have the section
    # but still have a parsable DELTA G.
    if "final_gbsa_delta_g_mean" not in results:
        # This is the old logic from the original function
        m_total = re.search(
            r"^\s*DELTA\s+TOTAL\s+([+-]?(?:\d+\.?\d*|\.\d+)(?:[Ee][+-]?\d+)?)\s+([+-]?(?:\d+\.?\d*|\.\d+)(?:[Ee][+-]?\d+)?)\s+([+-]?(?:\d+\.?\d*|\.\d+)(?:[Ee][+-]?\d+)?)",
            txt,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if m_total:
            results["final_gbsa_delta_g_mean"] = float(m_total.group(1))
            results["final_gbsa_delta_g_std"] = float(m_total.group(2))
            results["final_gbsa_delta_g_sem"] = float(m_total.group(3))
        else:
            m = re.search(
                r"DELTA\s+G(?:\s+binding)?\s*=\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[Ee][+-]?\d+)?)\s*\+/-\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[Ee][+-]?\d+)?)",
                txt,
                flags=re.IGNORECASE,
            )
            if m:
                results["final_gbsa_delta_g_mean"] = float(m.group(1))
                results["final_gbsa_delta_g_std"] = float(m.group(2))
            else:
                for raw in txt.splitlines():
                    up = raw.upper()
                    if ("DELTA G" not in up) or ("GAS" in up) or ("SOLV" in up):
                        continue
                    nums = re.findall(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[Ee][+-]?\d+)?", raw)
                    if len(nums) >= 2:
                        try:
                            results["final_gbsa_delta_g_mean"] = float(nums[0])
                            results["final_gbsa_delta_g_std"] = float(nums[1])
                            if len(nums) >= 3:
                                results["final_gbsa_delta_g_sem"] = float(nums[2])
                            break
                        except ValueError:
                            continue

    return results if results else None


def build_color_map(systems: List[str], reference: str) -> Dict[str, str]:
    # Keep reference in Dark Carbon Gray; others follow the new palette.
    other_palette = [
        '#1F77B4', '#D62728', '#2CA02C', '#FF7F0E',
        '#9467BD', '#8C564B', '#17BECF', '#E377C2'
    ]
    cmap: Dict[str, str] = {}
    cmap[reference] = "#333333"
    j = 0
    for s in systems:
        if s == reference:
            continue
        cmap[s] = other_palette[j % len(other_palette)]
        j += 1
    return cmap


def plot_multi(
    metric_name: str,
    ylabel: str,
    data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    out_svg: Path,
    reference: str,
    xlabel: str = "Time (ns)",
) -> None:
    if not data:
        return
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Cambria", "Times New Roman", "DejaVu Serif"],
            "font.size": 12.0,
            "axes.labelsize": 16.0,
            "xtick.labelsize": 14.0,
            "ytick.labelsize": 14.0,
            "axes.grid": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.top": False,
            "ytick.right": False,
            "svg.fonttype": "none",
        }
    )
    fig, ax = plt.subplots(figsize=(7.6, 4.6), constrained_layout=True)
    systems = sorted(data.keys())
    color_map = build_color_map(systems, reference)
    ordered = [reference] + [s for s in systems if s != reference] if reference in data else systems
    x_all = []
    y_all = []
    for sys_name in ordered:
        x, y = data[sys_name]
        lw = 1.6 if sys_name == reference else 1.25
        alpha = 0.95 if sys_name == reference else 0.9
        ax.plot(x, y, lw=lw, alpha=alpha, label=sys_name.upper(), color=color_map[sys_name])
        x_all.append(np.asarray(x, dtype=float))
        y_all.append(np.asarray(y, dtype=float))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if x_all and y_all:
        xv = np.concatenate(x_all)
        yv = np.concatenate(y_all)
        xv = xv[np.isfinite(xv)]
        yv = yv[np.isfinite(yv)]
        if xv.size:
            ax.set_xlim(float(np.min(xv)), float(np.max(xv)))
        if yv.size:
            y_min = float(np.min(yv))
            y_max = float(np.max(yv))
            if y_max == y_min:
                y_max = y_min + 1.0
            ax.set_ylim(y_min, y_max)
    ax.margins(x=0, y=0)
    ax.set_title(metric_name)
    ax.legend(fontsize=10, frameon=True, ncol=2, alignment="center")
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_multi_grid(
    metric_name: str,
    ylabel: str,
    data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    out_svg: Path,
    reference: str,
    nrows: int = 3,
    ncols: int = 3,
    xlabel: str = "Time (ns)",
    grid_legend_no_box: bool = False,
) -> None:
    if not data:
        return
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Cambria", "Times New Roman", "DejaVu Serif"],
            "font.size": 12.0,
            "axes.labelsize": 16.0,
            "xtick.labelsize": 14.0,
            "ytick.labelsize": 14.0,
            "axes.grid": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.top": False,
            "ytick.right": False,
            "svg.fonttype": "none",
        }
    )
    systems = sorted(data.keys())
    color_map = build_color_map(systems, reference)
    ordered = [reference] + [s for s in systems if s != reference] if reference in data else systems
    
    x_all = []
    y_all = []
    for sys_name in ordered:
        x, y = data[sys_name]
        x_all.append(np.asarray(x, dtype=float))
        y_all.append(np.asarray(y, dtype=float))
        
    if not x_all or not y_all:
        return
        
    xv = np.concatenate(x_all)
    yv = np.concatenate(y_all)
    xv = xv[np.isfinite(xv)]
    yv = yv[np.isfinite(yv)]
    
    x_min, x_max = 0.0, 1.0
    if xv.size:
        x_min, x_max = float(np.min(xv)), float(np.max(xv))
        
    y_min, y_max = 0.0, 1.0
    if yv.size:
        y_min = float(np.min(yv))
        y_max = float(np.max(yv))
        if y_max == y_min:
            y_max = y_min + 1.0
            
    y_pad = (y_max - y_min) * 0.1
    y_min = max(0.0, y_min - y_pad)
    y_max = y_max + y_pad

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 10), constrained_layout=True, dpi=600)
    axes = np.atleast_2d(axes)
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    for idx in range(nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r, c]
        
        if idx < len(ordered):
            sys_name = ordered[idx]
            x, y = data[sys_name]
            color = color_map[sys_name]
            
            ax.plot(x, y, lw=1.2, alpha=0.85, color=color)
            
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            
            # Panel Letter top left
            ax.text(
                0.03, 0.95,
                f"({letters[idx]})",
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=14, fontweight="bold",
                color="black"
            )
            
            # System Name top right
            sys_up = sys_name.upper()
            text_props = {
                "transform": ax.transAxes,
                "ha": "right", "va": "top",
                "fontsize": 12, "fontweight": "bold",
                "color": "black",
            }
            if not grid_legend_no_box:
                text_props["bbox"] = dict(boxstyle="round,pad=0.1", fc="white", ec="black", lw=1.2, alpha=0.9)
            ax.text(
                0.97, 0.95,
                sys_up,
                **text_props
            )
            
            if r == nrows - 1 or idx == len(ordered) - 1:
                ax.set_xlabel(xlabel, fontweight="bold", fontsize=14)
            if c == 0:
                ax.set_ylabel(ylabel, fontweight="bold", fontsize=14)
            
            for sp in ax.spines.values():
                sp.set_linewidth(1.2)
                sp.set_color("black")
            ax.tick_params(direction="out", length=4, width=1.2, top=False, right=False, labelsize=14)
        else:
            ax.set_visible(False)
            
    fig.suptitle(f"{metric_name} (3x3 Faceted by System)", fontsize=18, fontweight="bold")
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_svg, format="svg", bbox_inches="tight", dpi=600)
    plt.close(fig)


def plot_hbond_faceted(
    data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    out_svg: Path,
    reference: str,
) -> None:
    if not data:
        return
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Cambria", "Times New Roman", "DejaVu Serif"],
            "font.size": 11.0,
            "axes.labelsize": 14.0,
            "xtick.labelsize": 12.0,
            "ytick.labelsize": 12.0,
            "axes.grid": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.top": False,
            "ytick.right": False,
            "svg.fonttype": "none",
        }
    )
    systems = sorted(data.keys())
    ordered = [reference] + [s for s in systems if s != reference] if reference in data else systems
    cmap = build_color_map(ordered, reference)
    n = len(ordered)
    fig_h = max(1.3 * n, 4.5)
    fig, axes = plt.subplots(n, 1, figsize=(10.5, fig_h), sharex=True, constrained_layout=True)
    if n == 1:
        axes = [axes]

    # Global x/y range for consistent panels.
    x_all = np.concatenate([np.asarray(data[s][0], dtype=float) for s in ordered])
    y_all = np.concatenate([np.asarray(data[s][1], dtype=float) for s in ordered])
    x_all = x_all[np.isfinite(x_all)]
    y_all = y_all[np.isfinite(y_all)]
    x_min = float(np.min(x_all)) if x_all.size else 0.0
    x_max = float(np.max(x_all)) if x_all.size else 1.0
    y_max = float(np.max(y_all)) if y_all.size else 1.0
    y_top = max(1.0, np.ceil(y_max))

    for ax, sys_name in zip(axes, ordered):
        x, y = data[sys_name]
        color = cmap.get(sys_name, "#2B5C8A")
        # Dense vertical style similar to per-frame counts, but separated by facets.
        ax.vlines(x, 0.0, y, color=color, alpha=0.78, linewidth=0.9)
        ax.fill_between(x, 0.0, y, color=color, alpha=0.18, linewidth=0.0)
        ax.set_ylim(0.0, y_top)
        ax.set_ylabel("HBonds")
        # Show denser numeric ticks (not only 0 and max).
        ax.yaxis.set_major_locator(MultipleLocator(1.0))
        ax.margins(x=0, y=0)
        ax.text(
            1.005,
            0.85,
            sys_name.upper(),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            color="black",
        )
        # Keep panels clean and separated.
        for spine in ax.spines.values():
            spine.set_linewidth(0.9)
        ax.grid(False)
    axes[-1].set_xlim(x_min, x_max)
    axes[-1].set_xlabel("Time (ns)")
    fig.suptitle("09 Protein-Ligand H-Bonds (Faceted by System)", fontsize=14)
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_pca_cartoon_grid(
    run_dirs: List[Path],
    out_svg: Path,
    reference: str,
    nrows: int = 3,
    ncols: int = 3,
    grid_legend_no_box: bool = False,
) -> None:
    # Prefer PNG (raster rendered by PyMOL). Fallback to SVG if PNG missing is skipped.
    items: List[Tuple[str, Path]] = []
    for run_dir in run_dirs:
        sys_name = run_dir.name
        p_png = run_dir / "analysis" / "plots" / "16_PCA_Mode_Cartoon_Transition.png"
        p_svg = run_dir / "analysis" / "plots" / "16_PCA_Mode_Cartoon_Transition.svg"
        if p_png.exists():
            items.append((sys_name, p_png))
        elif p_svg.exists():
            # Matplotlib cannot reliably raster-read SVG here; skip if no PNG.
            continue
    if not items:
        return

    # Order: reference first, then others alphabetical.
    ordered: List[Tuple[str, Path]] = []
    ref = [x for x in items if x[0] == reference]
    others = sorted([x for x in items if x[0] != reference], key=lambda t: t[0])
    ordered.extend(ref)
    ordered.extend(others)

    max_panels = nrows * ncols
    ordered = ordered[:max_panels]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Cambria", "Times New Roman", "DejaVu Serif"],
            "font.size": 12.0,
            "axes.labelsize": 16.0,
            "xtick.labelsize": 14.0,
            "ytick.labelsize": 14.0,
            "axes.grid": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.top": False,
            "ytick.right": False,
            "svg.fonttype": "none",
        }
    )

    fig = plt.figure(figsize=(14.6, 12.2), facecolor="white", constrained_layout=True)
    # Reserve one bottom row-height slice for legend strip.
    outer = fig.add_gridspec(2, 1, height_ratios=[16.5, 1.5])
    grid = outer[0].subgridspec(nrows, ncols, wspace=0.006, hspace=0.045)

    for idx in range(max_panels):
        r = idx // ncols
        c = idx % ncols
        ax = fig.add_subplot(grid[r, c])
        if idx < len(ordered):
            sys_name, img_path = ordered[idx]
            try:
                img = plt.imread(img_path)
                ax.imshow(img)
                # Boxed system label inside each panel.
                text_props = {
                    "transform": ax.transAxes,
                    "ha": "left",
                    "va": "top",
                    "fontsize": 12,
                    "fontweight": "bold",
                    "color": "black",
                }
                if not grid_legend_no_box:
                    text_props["bbox"] = {"boxstyle": "round,pad=0.05", "fc": "white", "ec": "#333333", "lw": 0.8, "alpha": 0.95}
                ax.text(
                    0.03,
                    0.94,
                    sys_name.upper(),
                    **text_props,
                )
            except Exception:
                ax.text(0.5, 0.5, f"{sys_name}\n(image read failed)", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(True)
        for sp in ax.spines.values():
            sp.set_linewidth(0.8)
            sp.set_color("#444444")

    # Legend strip (0 ns -> 200 ns)
    lax = fig.add_subplot(outer[1])
    lax.set_xlim(0, 1)
    lax.set_ylim(0, 1)
    lax.axis("off")
    c0 = "#2f65d9"   # start color used in pymol script
    c1 = "#c07a1a"   # end color used in pymol script
    left = Circle((0.22, 0.52), 0.08, facecolor=c0, edgecolor="none", alpha=0.95)
    right = Circle((0.78, 0.52), 0.08, facecolor=c1, edgecolor="none", alpha=0.95)
    lax.add_patch(left)
    lax.add_patch(right)
    arr = FancyArrowPatch((0.30, 0.52), (0.70, 0.52), arrowstyle="-", linewidth=2.2, color="#8ea0bf", alpha=0.95)
    lax.add_patch(arr)
    lax.text(0.22, 0.22, "0 ns", ha="center", va="center", fontsize=14, fontweight="bold", color="black")
    lax.text(0.78, 0.22, "200 ns", ha="center", va="center", fontsize=14, fontweight="bold", color="black")
    lax.text(0.50, 0.82, "Transition Color Legend", ha="center", va="center", fontsize=11)

    fig.suptitle("16 PCA Mode Cartoon Transition (3×3 Systems)", fontsize=16)
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_single(
    metric_name: str,
    ylabel: str,
    x: np.ndarray,
    y: np.ndarray,
    out_svg: Path,
    color: str,
    system: str,
) -> None:
    if x.size == 0 or y.size == 0:
        return
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Cambria", "Times New Roman", "DejaVu Serif"],
            "font.size": 12.0,
            "axes.labelsize": 16.0,
            "xtick.labelsize": 14.0,
            "ytick.labelsize": 14.0,
            "axes.grid": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.top": False,
            "ytick.right": False,
            "svg.fonttype": "none",
        }
    )
    fig, ax = plt.subplots(figsize=(7.0, 4.3), constrained_layout=True)
    ax.plot(x, y, lw=1.5, alpha=0.92, color=color, label=system.upper())
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel(ylabel)
    xv = x[np.isfinite(x)]
    yv = y[np.isfinite(y)]
    if xv.size:
        ax.set_xlim(float(np.min(xv)), float(np.max(xv)))
    if yv.size:
        y_min = float(np.min(yv))
        y_max = float(np.max(yv))
        if y_max == y_min:
            y_max = y_min + 1.0
        ax.set_ylim(y_min, y_max)
    ax.margins(x=0, y=0)
    ax.set_title(f"{metric_name} ({system.upper()})")
    ax.legend(fontsize=10, frameon=True, alignment="center")
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_rmsf_single_with_key_residues(
    residue_idx: np.ndarray,
    rmsf: np.ndarray,
    out_svg: Path,
    color: str,
    system: str,
    residue_label_map: Optional[Dict[int, str]] = None,
    topn: int = 8,
) -> None:
    if residue_idx.size == 0 or rmsf.size == 0:
        return
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Cambria", "Times New Roman", "DejaVu Serif"],
            "font.size": 12.0,
            "axes.labelsize": 16.0,
            "xtick.labelsize": 14.0,
            "ytick.labelsize": 14.0,
            "axes.grid": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.top": False,
            "ytick.right": False,
            "svg.fonttype": "none",
        }
    )
    fig, ax = plt.subplots(figsize=(7.0, 4.3), constrained_layout=True)
    ax.plot(residue_idx, rmsf, lw=1.35, alpha=0.95, color=color, label=system.upper())
    ax.set_xlabel("Residue Index")
    ax.set_ylabel("RMSF (Å)")
    ax.set_title(f"02 RMSF Profile ({system.upper()})")
    xv = residue_idx[np.isfinite(residue_idx)]
    yv = rmsf[np.isfinite(rmsf)]
    if xv.size:
        ax.set_xlim(float(np.min(xv)), float(np.max(xv)))
    if yv.size:
        y_min = float(np.min(yv))
        y_max = float(np.max(yv))
        if y_max == y_min:
            y_max = y_min + 1.0
        pad = 0.18 * (y_max - y_min)
        ax.set_ylim(y_min - pad, y_max + pad)
    ax.margins(x=0, y=0)

    # Key residues: annotate top-N RMSF peaks per system.
    n_use = min(int(topn), int(rmsf.size))
    if n_use > 0:
        order = np.argsort(rmsf)[::-1]
        keep = np.sort(order[:n_use])
        offsets = [(-16, 14), (16, 14), (-14, -12), (14, -12), (-20, 10), (20, 10), (-10, 16), (10, 16)]
        x_min = float(np.min(residue_idx))
        x_max = float(np.max(residue_idx))
        keep_list = keep.tolist()
        right_zone = x_max - 20.0
        right_idxs = [k for k in keep_list if float(residue_idx[k]) >= right_zone]
        # Deterministic left-stagger for crowded right-edge labels.
        right_rank = {k: r for r, k in enumerate(sorted(right_idxs, key=lambda t: float(residue_idx[t]), reverse=True))}
        placed: List[Tuple[float, float]] = []
        for j, idx in enumerate(keep_list):
            rx = float(residue_idx[idx])
            ry = float(rmsf[idx])
            rid = int(round(rx))
            rlab = residue_label_map.get(rid, f"RES{rid}") if residue_label_map else f"RES{rid}"
            ax.scatter([rx], [ry], s=16, color="#d62728", zorder=3)
            dx, dy = offsets[j % len(offsets)]
            # Push edge residues inward so labels stay inside panel.
            if rx <= x_min + 8:
                dx = abs(dx) + 8
            elif rx >= x_max - 8:
                dx = -abs(dx) - 8
            # Strong deterministic staggering for right-edge crowded labels.
            if idx in right_rank:
                r = right_rank[idx]
                dx = -24 - 22 * r
                dy = 18 if (r % 2 == 0) else -16
            # If two key residues are close in x, force stronger stagger to avoid overlap.
            for px, py in placed:
                if abs(rx - px) <= 6 and idx not in right_rank:
                    if rx >= x_max - 20:
                        dx -= 16
                    elif rx <= x_min + 20:
                        dx += 16
                    dy = 20 if ry <= py else -16
            placed.append((rx, ry))
            ax.annotate(
                f"{rlab}{rid}",
                xy=(rx, ry),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#8b0000",
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#8b0000", "lw": 0.7, "alpha": 0.95},
                arrowprops={"arrowstyle": "-", "lw": 0.6, "color": "#8b0000", "alpha": 0.8},
            )

    ax.legend(fontsize=10, frameon=True, alignment="center")
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_zn_violin(single_data: Dict[str, np.ndarray], cluster_data: Dict[str, np.ndarray], reference: str, out_svg: Path) -> None:
    if not single_data and not cluster_data:
        return
    try:
        import seaborn as sns
    except ImportError:
        print("[WARN] seaborn not installed, skipping Zn violin plot")
        return

    import pandas as pd
    systems = sorted(list(set(list(single_data.keys()) + list(cluster_data.keys()))))
    ordered = [reference] + [s for s in systems if s != reference] if reference in systems else systems

    rows = []
    for sys in ordered:
        if sys in single_data:
            for val in single_data[sys]:
                rows.append({"System": sys.upper(), "Distance (Å)": val, "Site": "Single (Zn221)"})
        if sys in cluster_data:
            for val in cluster_data[sys]:
                rows.append({"System": sys.upper(), "Distance (Å)": val, "Site": "Cluster (Zn222/Zn223)"})

    if not rows:
        return
    df = pd.DataFrame(rows)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Cambria", "Times New Roman", "DejaVu Serif"],
            "font.size": 14.0,
            "axes.labelsize": 16.0,
            "xtick.labelsize": 14.0,
            "ytick.labelsize": 14.0,
            "axes.grid": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.top": False,
            "ytick.right": False,
            "svg.fonttype": "none",
        }
    )

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    sns.violinplot(
        data=df, x="System", y="Distance (Å)", hue="Site",
        inner="quart", ax=ax, palette={"Single (Zn221)": "#4A6B82", "Cluster (Zn222/Zn223)": "#B85A4B"},
        linewidth=1.0, dodge=True, cut=0
    )

    ax.set_title("Zn Coordination Distance Distribution (Last 50 ns)", fontsize=18, fontweight="bold", pad=12)
    ax.set_xlabel("System", fontsize=16, fontweight="bold")
    ax.set_ylabel("Distance (Å)", fontsize=16, fontweight="bold")
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=14)

    y_min = max(0, df["Distance (Å)"].min() - 0.1)
    y_max = df["Distance (Å)"].max() + 0.1
    ax.set_ylim(y_min, y_max)

    ax.legend(title="Coordination Site", frameon=True, edgecolor="black", loc="upper right", fontsize=13, title_fontsize=14)

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("black")

    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    plt.close(fig)

    # --- Generate Horizontal Version ---
    out_svg_h = out_svg.parent / f"{out_svg.stem}_horizontal{out_svg.suffix}"
    fig_h, ax_h = plt.subplots(figsize=(7.0, max(5.5, len(ordered) * 0.6)), constrained_layout=True)
    
    sns.violinplot(
        data=df, x="Distance (Å)", y="System", hue="Site",
        inner="quart", ax=ax_h, palette={"Single (Zn221)": "#4A6B82", "Cluster (Zn222/Zn223)": "#B85A4B"},
        linewidth=1.0, dodge=True, orient="h", cut=0
    )

    ax_h.set_title("Zn Coordination Distance Distribution (Last 50 ns)", fontsize=18, fontweight="bold", pad=12)
    ax_h.set_ylabel("System", fontsize=16, fontweight="bold")
    ax_h.set_xlabel("Distance (Å)", fontsize=16, fontweight="bold")
    
    plt.setp(ax_h.get_yticklabels(), fontsize=14)
    plt.setp(ax_h.get_xticklabels(), fontsize=14)

    ax_h.set_xlim(y_min, y_max)

    ax_h.legend(title="Coordination Site", frameon=True, edgecolor="black", loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=13, title_fontsize=14)

    for spine in ax_h.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("black")

    fig_h.savefig(out_svg_h, format="svg", bbox_inches="tight")
    plt.close(fig_h)

def plot_zn_distances_single_system(zn_files: List[Path], out_svg: Path, dt: float, ntwx: int, stride: int, system_name: str) -> None:
    if not zn_files:
        return
    
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Cambria", "Times New Roman", "DejaVu Serif"],
        "font.size": 12.0,
        "axes.labelsize": 14.0,
        "xtick.labelsize": 12.0,
        "ytick.labelsize": 12.0,
        "axes.grid": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.top": False,
        "ytick.right": False,
        "svg.fonttype": "none",
    })

    colors = {
        "ZN221_CYM161_ZN_SG.dat": "#2c3e50", "ZN221_CYM208_ZN_SG.dat": "#e74c3c",
        "ZN221_CYM210_ZN_SG.dat": "#27ae60", "ZN221_CYM215_ZN_SG.dat": "#f39c12",
        "ZN222_CYM33_ZN_SG.dat": "#2980b9", "ZN222_CYM35_ZN_SG.dat": "#8e44ad",
        "ZN222_CYM43_ZN_SG.dat": "#c0392b", "ZN222_CYM49_ZN_SG.dat": "#16a085",
        "ZN223_CYM43_ZN_SG.dat": "#d35400", "ZN223_CYM58_ZN_SG.dat": "#7f8c8d",
        "ZN223_CYM63_ZN_SG.dat": "#2c3e50", "ZN223_CYM69_ZN_SG.dat": "#bdc3c7"
    }

    group1 = [f for f in zn_files if "ZN221" in f.name]
    group2 = [f for f in zn_files if "ZN222" in f.name]
    group3 = [f for f in zn_files if "ZN223" in f.name]

    if not group1 and not group2 and not group3:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True, constrained_layout=True)

    def plot_zn_group(ax, data_paths, title):
        all_dist = []
        for path in data_paths:
            data = read_table(path)
            if data is None or data.shape[1] < 2:
                continue
            time = (data[:, 0].astype(float) * ntwx * stride * dt) / 1000.0
            dist = data[:, 1].astype(float)
            mask = np.isfinite(time) & np.isfinite(dist)
            time = time[mask]
            dist = dist[mask]
            if dist.size:
                all_dist.append(dist)
            
            label = path.stem.replace("ZN", "Zn").replace("_ZN_SG", "")
            color = colors.get(path.name, '#7f8c8d')
            
            ax.plot(time, dist, lw=0.3, color=color, alpha=0.2)
            
            window = max(1, len(dist) // 50)
            if window > 1:
                dist_smooth = np.convolve(dist, np.ones(window), 'valid') / window
                time_smooth = time[window-1:]
                ax.plot(time_smooth, dist_smooth, lw=1.5, color=color, label=label)
            else:
                ax.plot(time, dist, lw=1.5, color=color, label=label)

        ax.set_title(title, loc='left', fontsize=12, fontweight='bold')
        ax.set_ylabel("Distance (Å)", fontsize=13)
        if all_dist:
            y = np.concatenate(all_dist)
            if y.size > 0:
                y_min = float(np.min(y))
                y_max = float(np.max(y))
                span = max(y_max - y_min, 0.15)
                low = max(0, y_min - 0.1 * span)
                high = y_max + 0.1 * span
                ax.set_ylim(low, high)
        
        ax.margins(x=0)
        if data_paths:
            ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10, frameon=True, edgecolor='black')

        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color("black")
            
        ax.tick_params(direction="out", length=4, width=1.2, top=False, right=False)

    plot_zn_group(ax1, group1, f"A: Zn221 Coordination (Single Site) - {system_name.upper()}")
    plot_zn_group(ax2, group2 + group3, f"B: Zn222 & Zn223 Cluster (Shared Site) - {system_name.upper()}")

    ax2.set_xlabel("Time (ns)", fontsize=13)
    
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_svg, format="svg", bbox_inches='tight')
    plt.close(fig)


def safe_copy(src: Path, dst_dir: Path) -> None:
    if src.exists():
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / src.name)


def copy_glob(src_dir: Path, patterns: List[str], dst_dir: Path) -> None:
    if not src_dir.exists():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    for pat in patterns:
        for fp in sorted(src_dir.glob(pat)):
            if fp.is_file():
                shutil.copy2(fp, dst_dir / fp.name)


def parse_residue_label_map(run_dir: Path) -> Dict[int, str]:
    # Try protein_clean.pdb first; fallback to complex_No_WAT.pdb.
    candidates = [run_dir / "protein_clean.pdb", run_dir / "complex_No_WAT.pdb"]
    pdb_path = pick_first_existing(candidates)
    if pdb_path is None:
        return {}
    mapping: Dict[int, str] = {}
    try:
        with pdb_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not (line.startswith("ATOM") or line.startswith("HETATM")):
                    continue
                if len(line) < 26:
                    continue
                resname = line[17:20].strip()
                resseq_txt = line[22:26].strip()
                if not resname or not resseq_txt:
                    continue
                try:
                    resseq = int(resseq_txt)
                except ValueError:
                    continue
                if resseq not in mapping:
                    mapping[resseq] = resname
    except Exception:
        return {}
    return mapping


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate/check MD metrics across systems.")
    ap.add_argument("command", nargs="?", default="aggregate", choices=["aggregate", "check"])
    ap.add_argument("--runs-root", type=Path, default=Path("mds/runs"))
    ap.add_argument("--outdir", type=Path, default=Path("mds/aggregate_outputs"))
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--ntwx", type=int, default=5000)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--reference", type=str, default="a1a0m", help="Reference system highlighted in light gray.")
    ap.add_argument("--grid-legend-no-box", action="store_true", help="Remove box from system labels in 3x3 grid plots.")
    ap.add_argument("--plot-zn-violin", action="store_true", help="Generate violin plot comparing Zn coordination distances.")
    args = ap.parse_args()

    if args.command == "check":
        runs_root = args.runs_root.resolve()
        required = {
            "01_QC": ["analysis/QC_Calpha_RMSD.dat"],
            "02_RMSD-RMSF": ["analysis/Calpha_RMSD.dat", "analysis/RMSF.dat"],
            "03_Ligand-RMSD": ["analysis/Ligand_RMSD.dat"],
            "06_RoG": ["analysis/RoG_Calpha.dat"],
            "07_SASA": ["analysis/SASA_protein.dat", "analysis/SASA_ligand.dat"],
            "09_HBond-Complex": ["analysis/HBond_PL_p2l.hbvtime.dat", "analysis/HBond_PL_l2p.hbvtime.dat"],
            "16_FEL": ["analysis/PCA_projection.dat", "analysis/plots/16_FEL_PC1_PC2_contour.svg"],
            "27_MMPBSA-GBSA": [
                "MMGBSA_vs_time.dat",
                "MMGBSA_vs_time_last50ns.dat",
                "MMGBSA_summary.csv",
                "MMGBSA_summary_last50ns.csv",
                "FINAL_GBSA.dat",
            ],
            "zn_distances": ["analysis/ZN221_CYM161_ZN_SG.dat", "analysis/plots/00_Zn_Coordination_Stability.svg"],
        }
        any_missing = False
        run_dirs = sorted([p for p in runs_root.iterdir() if p.is_dir()])
        for run_dir in run_dirs:
            sys_name = run_dir.name
            missing_items = []
            for block, rels in required.items():
                miss = [r for r in rels if not (run_dir / r).exists()]
                if miss:
                    missing_items.append((block, miss))
            if missing_items:
                any_missing = True
                print(f"[MISSING] {sys_name}")
                for block, miss in missing_items:
                    print(f"  - {block}")
                    for m in miss:
                        print(f"    * {m}")
            else:
                print(f"[OK] {sys_name}")
        if any_missing:
            print("[SUMMARY] Some systems are missing baseline outputs.")
        else:
            print("[SUMMARY] All systems have baseline outputs.")
        return

    runs_root = args.runs_root.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    metrics: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {
        "02_Calpha_RMSD": {},
        "03_Ligand_RMSD": {},
        "06_RoG": {},
        "07_SASA_protein": {},
        "07_SASA_ligand": {},
        "09_HBond_total": {},
    }
    rmsf_profiles: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    residue_maps: Dict[str, Dict[int, str]] = {}
    summary_rows: List[Dict[str, object]] = []
    ts_rows: List[Dict[str, object]] = []
    key_rows: List[Dict[str, object]] = []
    final_gbsa_rows: List[Dict[str, object]] = []
    per_system_summary: Dict[str, List[Dict[str, object]]] = {}
    per_system_ts: Dict[str, List[Dict[str, object]]] = {}
    per_system_metric_file: Dict[str, Dict[str, Path]] = {}
    per_system_zn_single: Dict[str, np.ndarray] = {}
    per_system_zn_cluster: Dict[str, np.ndarray] = {}

    run_dirs = sorted([p for p in runs_root.iterdir() if p.is_dir()])
    all_systems = [p.name for p in run_dirs]
    global_cmap = build_color_map(all_systems, args.reference)
    for run_dir in run_dirs:
        sys_name = run_dir.name
        analysis_dir = run_dir / "analysis"
        if not analysis_dir.exists():
            continue
        residue_maps[sys_name] = parse_residue_label_map(run_dir)
        per_system_summary.setdefault(sys_name, [])
        per_system_ts.setdefault(sys_name, [])
        per_system_metric_file.setdefault(sys_name, {})

        file_candidates = {
            "02_Calpha_RMSD": [analysis_dir / "Calpha_RMSD.dat"],
            "03_Ligand_RMSD": [analysis_dir / "Ligand_RMSD.dat", analysis_dir / "Ligand-RMSD.dat"],
            "06_RoG": [analysis_dir / "RoG_Calpha.dat", analysis_dir / "RoG.dat"],
            "07_SASA_protein": [analysis_dir / "SASA_protein.dat"],
            "07_SASA_ligand": [analysis_dir / "SASA_ligand.dat"],
        }
        print(f"[SCAN] {sys_name}")
        key_row: Dict[str, object] = {"system": sys_name}

        for metric, cand in file_candidates.items():
            path = pick_first_existing(cand)
            if path is None:
                print(f"  - missing {metric}: {[str(x.name) for x in cand]}")
                continue
            print(f"  - {metric}: {path.name}")
            xy = extract_xy(path, dt=args.dt, ntwx=args.ntwx, stride=args.stride, ycol=1)
            if xy is None:
                print(f"    ! parse failed: {path.name}")
                continue
            x, y = xy
            metrics[metric][sys_name] = (x, y)
            per_system_metric_file[sys_name][metric] = path
            stats = summarize(y)
            summary_rows.append({"system": sys_name, "metric": metric, **stats})
            per_system_summary[sys_name].append({"system": sys_name, "metric": metric, **stats})
            for xi, yi in zip(x, y):
                ts_rows.append({"system": sys_name, "metric": metric, "time_ns": float(xi), "value": float(yi)})
                per_system_ts[sys_name].append({"system": sys_name, "metric": metric, "time_ns": float(xi), "value": float(yi)})

            # Paper-friendly equilibrium window summary (last 50 ns ~= last 25 points in current setup).
            last_stats = summarize_last_n(y, n_last=25)
            key_row[f"{metric}_last50ns_mean"] = last_stats["mean"]
            key_row[f"{metric}_last50ns_sd"] = last_stats["std"]

        # HBond total = p2l + l2p
        p2l_path = analysis_dir / "HBond_PL_p2l.hbvtime.dat"
        l2p_path = analysis_dir / "HBond_PL_l2p.hbvtime.dat"
        p2l = extract_xy(p2l_path, dt=args.dt, ntwx=args.ntwx, stride=args.stride, ycol=1)
        l2p = extract_xy(l2p_path, dt=args.dt, ntwx=args.ntwx, stride=args.stride, ycol=1)
        if p2l is not None and l2p is not None:
            print("  - 09_HBond_total: HBond_PL_p2l.hbvtime.dat + HBond_PL_l2p.hbvtime.dat")
            x1, y1 = p2l
            x2, y2 = l2p
            n = min(len(x1), len(x2))
            x = x1[:n]
            y = y1[:n] + y2[:n]
            metrics["09_HBond_total"][sys_name] = (x, y)
            per_system_metric_file[sys_name]["09_HBond_total"] = analysis_dir / "HBond_PL_total.synthetic.dat"
            stats = summarize(y)
            summary_rows.append({"system": sys_name, "metric": "09_HBond_total", **stats})
            per_system_summary[sys_name].append({"system": sys_name, "metric": "09_HBond_total", **stats})
            for xi, yi in zip(x, y):
                ts_rows.append({"system": sys_name, "metric": "09_HBond_total", "time_ns": float(xi), "value": float(yi)})
                per_system_ts[sys_name].append({"system": sys_name, "metric": "09_HBond_total", "time_ns": float(xi), "value": float(yi)})
            last_stats = summarize_last_n(y, n_last=25)
            key_row["09_HBond_total_last50ns_mean"] = last_stats["mean"]
            key_row["09_HBond_total_last50ns_sd"] = last_stats["std"]
        else:
            print("  - missing/invalid 09_HBond_total inputs")

        # RMSF: Read profile for both multi-plot and single-plot.
        rmsf_path = analysis_dir / "RMSF.dat"
        rmsf_arr = read_table(rmsf_path)
        if rmsf_arr is not None and rmsf_arr.shape[1] >= 2:
            rx = rmsf_arr[:, 0].astype(float)
            ry = rmsf_arr[:, 1].astype(float)
            mask = np.isfinite(rx) & np.isfinite(ry)
            if np.any(mask):
                rmsf_profiles[sys_name] = (rx[mask], ry[mask])
                print("  - RMSF profile: RMSF.dat")
            else:
                print("  - RMSF profile invalid: RMSF.dat")
        else:
            print("  - missing RMSF.dat")

        mmgbsa_last50 = read_mmgbsa_last50_summary(run_dir / "MMGBSA_summary_last50ns.csv")
        if mmgbsa_last50 is not None:
            key_row.update(mmgbsa_last50)
            print("  - MMGBSA last50: MMGBSA_summary_last50ns.csv")
        else:
            print("  - missing MMGBSA_summary_last50ns.csv")

        hbond_top1 = read_hbond_top1(analysis_dir / "HBond_PL_occupancy_summary.csv")
        if hbond_top1 is not None:
            key_row.update(hbond_top1)
            print("  - HBond top1: HBond_PL_occupancy_summary.csv")
        else:
            print("  - missing HBond_PL_occupancy_summary.csv")

        final_g = parse_final_gbsa_delta_g(run_dir / "FINAL_GBSA.dat")
        if final_g is not None:
            key_row.update(final_g)

            gbsa_row = {"system": sys_name}
            gbsa_row.update(final_g)
            gbsa_row["source_file"] = str((run_dir / "FINAL_GBSA.dat").resolve())
            final_gbsa_rows.append(gbsa_row)

            print("  - FINAL_GBSA components: FINAL_GBSA.dat")
        else:
            print("  - missing/unparsed FINAL_GBSA.dat")

        # Process ZN files for the last 50ns
        zn_files = sorted(analysis_dir.glob("ZN*.dat"))
        if zn_files:
            print(f"  - ZN distances: found {len(zn_files)} files, adding to key_results")
            zn_means_for_overall = []
            single_raw = []
            cluster_raw = []
            for zn_file in zn_files:
                zn_stats = extract_zn_last_50ns_stats(zn_file, args.dt, args.ntwx, stride=1)
                if zn_stats is not None:
                    mean_val, std_val = zn_stats
                    key_row[f"{zn_file.stem}_last50ns_mean"] = mean_val
                    key_row[f"{zn_file.stem}_last50ns_std"] = std_val
                    zn_means_for_overall.append(mean_val)

                if args.plot_zn_violin:
                    raw_data = extract_zn_last_50ns_raw(zn_file, args.dt, args.ntwx, stride=1)
                    if raw_data is not None:
                        if "ZN221" in zn_file.name:
                            single_raw.append(raw_data)
                        elif "ZN222" in zn_file.name or "ZN223" in zn_file.name:
                            cluster_raw.append(raw_data)

            if zn_means_for_overall:
                key_row["ZN_overall_last50ns_mean"] = float(np.mean(zn_means_for_overall))
                key_row["ZN_overall_last50ns_std"] = float(np.std(zn_means_for_overall, ddof=1) if len(zn_means_for_overall) > 1 else 0.0)

            if args.plot_zn_violin:
                if single_raw:
                    per_system_zn_single[sys_name] = np.concatenate(single_raw)
                if cluster_raw:
                    per_system_zn_cluster[sys_name] = np.concatenate(cluster_raw)

        key_rows.append(key_row)

    plot_multi("02 Cα RMSD", "Cα RMSD (Å)", metrics["02_Calpha_RMSD"], outdir / "02_Calpha_RMSD_multi.svg", args.reference)
    plot_multi_grid("02 Cα RMSD", "Cα RMSD (Å)", metrics["02_Calpha_RMSD"], outdir / "02_Calpha_RMSD_grid_3x3.svg", args.reference, grid_legend_no_box=args.grid_legend_no_box)
    
    plot_multi("02 RMSF Profile", "RMSF (Å)", rmsf_profiles, outdir / "02_RMSF_multi.svg", args.reference, xlabel="Residue Index")
    plot_multi_grid("02 RMSF Profile", "RMSF (Å)", rmsf_profiles, outdir / "02_RMSF_grid_3x3.svg", args.reference, xlabel="Residue Index", grid_legend_no_box=args.grid_legend_no_box)
    
    plot_multi("03 Ligand RMSD", "Ligand RMSD (Å)", metrics["03_Ligand_RMSD"], outdir / "03_Ligand_RMSD_multi.svg", args.reference)
    plot_multi_grid("03 Ligand RMSD", "Ligand RMSD (Å)", metrics["03_Ligand_RMSD"], outdir / "03_Ligand_RMSD_grid_3x3.svg", args.reference, grid_legend_no_box=args.grid_legend_no_box)
    
    plot_multi("06 Radius of Gyration", "RoG (Å)", metrics["06_RoG"], outdir / "06_RoG_multi.svg", args.reference)
    plot_multi_grid("06 Radius of Gyration", "RoG (Å)", metrics["06_RoG"], outdir / "06_RoG_grid_3x3.svg", args.reference, grid_legend_no_box=args.grid_legend_no_box)
    
    plot_multi("07 Protein SASA", "Protein SASA (Å²)", metrics["07_SASA_protein"], outdir / "07_SASA_protein_multi.svg", args.reference)
    plot_multi_grid("07 Protein SASA", "Protein SASA (Å²)", metrics["07_SASA_protein"], outdir / "07_SASA_protein_grid_3x3.svg", args.reference, grid_legend_no_box=args.grid_legend_no_box)
    
    plot_multi("07 Ligand SASA", "Ligand SASA (Å²)", metrics["07_SASA_ligand"], outdir / "07_SASA_ligand_multi.svg", args.reference)
    plot_multi_grid("07 Ligand SASA", "Ligand SASA (Å²)", metrics["07_SASA_ligand"], outdir / "07_SASA_ligand_grid_3x3.svg", args.reference, grid_legend_no_box=args.grid_legend_no_box)
    
    plot_multi("09 Protein-Ligand H-Bonds", "Number of Hydrogen Bonds", metrics["09_HBond_total"], outdir / "09_HBond_total_multi.svg", args.reference)
    plot_hbond_faceted(metrics["09_HBond_total"], outdir / "09_HBond_total_faceted.svg", args.reference)
    plot_pca_cartoon_grid(run_dirs, outdir / "16_PCA_Mode_Cartoon_Transition_grid_3x3.svg", args.reference, nrows=3, ncols=3, grid_legend_no_box=args.grid_legend_no_box)

    if args.plot_zn_violin:
        plot_zn_violin(per_system_zn_single, per_system_zn_cluster, args.reference, outdir / "00_Zn_Coordination_Violin.svg")

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(outdir / "md_summary_all_systems.csv", index=False)
    if ts_rows:
        pd.DataFrame(ts_rows).to_csv(outdir / "md_timeseries_long.csv", index=False)
    if key_rows:
        pd.DataFrame(key_rows).to_csv(outdir / "md_key_results_last50ns.csv", index=False)
    if final_gbsa_rows:
        pd.DataFrame(final_gbsa_rows).to_csv(outdir / "final_gbsa_summary_all_systems.csv", index=False)

    # Per-system folders: split plots + copied key result files + per-system CSV.
    metric_plot_meta = {
        "02_Calpha_RMSD": ("02 Cα RMSD", "Cα RMSD (Å)", "02_Calpha_RMSD.svg"),
        "03_Ligand_RMSD": ("03 Ligand RMSD", "Ligand RMSD (Å)", "03_Ligand_RMSD.svg"),
        "06_RoG": ("06 Radius of Gyration", "RoG (Å)", "06_RoG.svg"),
        "07_SASA_protein": ("07 Protein SASA", "Protein SASA (Å²)", "07_SASA_protein.svg"),
        "07_SASA_ligand": ("07 Ligand SASA", "Ligand SASA (Å²)", "07_SASA_ligand.svg"),
        "09_HBond_total": ("09 Protein-Ligand H-Bonds", "Number of Hydrogen Bonds", "09_HBond_total.svg"),
    }
    for run_dir in run_dirs:
        sys_name = run_dir.name
        sys_dir = outdir / "systems" / sys_name
        sys_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = sys_dir / "plots"
        data_dir = sys_dir / "data"
        color = global_cmap.get(sys_name, "#2B5C8A")

        # Split plots using the same color mapping as combined plots.
        for metric, (title, ylabel, fname) in metric_plot_meta.items():
            if sys_name in metrics[metric]:
                x, y = metrics[metric][sys_name]
                plot_single(title, ylabel, x, y, plots_dir / fname, color=color, system=sys_name)
        if sys_name in rmsf_profiles:
            rx, ry = rmsf_profiles[sys_name]
            plot_rmsf_single_with_key_residues(
                rx,
                ry,
                plots_dir / "02_RMSF_Profile_annotated.svg",
                color=color,
                system=sys_name,
                residue_label_map=residue_maps.get(sys_name, {}),
                topn=8,
            )

        # Per-system CSV exports.
        if per_system_summary.get(sys_name):
            pd.DataFrame(per_system_summary[sys_name]).to_csv(sys_dir / "summary.csv", index=False)
        if per_system_ts.get(sys_name):
            pd.DataFrame(per_system_ts[sys_name]).to_csv(sys_dir / "timeseries.csv", index=False)

        # Copy key source/result files for quick packaging.
        analysis_dir = run_dir / "analysis"
        copy_list = [
            run_dir / "analysis_masks.txt",
            run_dir / "FINAL_GBSA.dat",
            run_dir / "MMGBSA_vs_time.dat",
            run_dir / "MMGBSA_vs_time_last50ns.dat",
            run_dir / "MMGBSA_summary.csv",
            run_dir / "MMGBSA_summary_last50ns.csv",
            run_dir / "_MMPBSA_info",
            run_dir / "_MMPBSA_complex_gb.mdout.0",
            run_dir / "_MMPBSA_receptor_gb.mdout.0",
            run_dir / "_MMPBSA_ligand_gb.mdout.0",
            run_dir / "_MMPBSA_complex_gb_surf.dat.0",
            run_dir / "_MMPBSA_receptor_gb_surf.dat.0",
            run_dir / "_MMPBSA_ligand_gb_surf.dat.0",
            analysis_dir / "Calpha_RMSD.dat",
            analysis_dir / "RMSF.dat",
            analysis_dir / "Ligand_RMSD.dat",
            analysis_dir / "RoG_Calpha.dat",
            analysis_dir / "SASA_protein.dat",
            analysis_dir / "SASA_ligand.dat",
            analysis_dir / "HBond_PL_p2l.hbvtime.dat",
            analysis_dir / "HBond_PL_l2p.hbvtime.dat",
            analysis_dir / "HBond_PL_p2l.avg.dat",
            analysis_dir / "HBond_PL_l2p.avg.dat",
            analysis_dir / "HBond_PL_occupancy_summary.csv",
        ]
        for fp in copy_list:
            safe_copy(fp, data_dir)

        # Copy Zn coordination analysis results
        copy_glob(analysis_dir, patterns=["ZN*"], dst_dir=data_dir)
        
        # Plot Zn distances for this system
        sys_zn_files = sorted(analysis_dir.glob("ZN*.dat"))
        plot_zn_distances_single_system(sys_zn_files, plots_dir / "00_Zn_Coordination_Stability.svg", args.dt, args.ntwx, args.stride, sys_name)

        # Copy publication-grade plots produced in each system's analysis/plots.
        sys_analysis_plots = analysis_dir / "plots"
        copy_glob(
            sys_analysis_plots,
            patterns=[
                "14_PCA_variance_contribution.svg",
                "14_PCA_PC1_PC2_timecolored.svg",
                "16_FEL_PC1_PC2_contour.svg",
                "16_FEL_PC1_PC2_surface3D.svg",
                "16_PCA_Mode_Cartoon_Transition.svg",
                "16_PCA_Mode_Cartoon_Transition_Panel.svg",
                "16_PCA_Mode_Cartoon_Transition.png",
            ],
            dst_dir=plots_dir,
        )

    print("[OK] Aggregation complete")
    print(f"  - Output dir: {outdir}")
    print("  - SVG: 02/03/06/07/09 multi-system plots (including RMSF) + grid_3x3 plots + 09_HBond_total_faceted.svg")
    print("  - SVG: 16_PCA_Mode_Cartoon_Transition_grid_3x3.svg")
    if args.plot_zn_violin:
        print("  - SVG: 00_Zn_Coordination_Violin.svg (and _horizontal.svg)")
    print("  - CSV: md_summary_all_systems.csv, md_timeseries_long.csv, md_key_results_last50ns.csv")
    print("  - CSV: final_gbsa_summary_all_systems.csv (from FINAL_GBSA.dat)")
    print("  - Per-system: systems/<system>/{plots,data,summary.csv,timeseries.csv}")
    print("  - In each system/data: HBond occupancy CSV + MMPBSA key files (if present)")


if __name__ == "__main__":
    main()
