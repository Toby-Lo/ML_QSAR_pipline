#!/usr/bin/env python3
"""
Generate manuscript-style PCA mode panel with real protein cartoon overlay.

Top panel:
  - PyMOL-rendered cartoon overlay from multiple sampled frames
    (blue -> orange transition; e.g., 0 ns -> 200 ns)
Bottom panel:
  - PCA explained-variance bar plot (top PCs) + cumulative line

Outputs:
  - analysis/plots/16_PCA_Mode_Cartoon_Transition.png
  - analysis/plots/16_PCA_Mode_Cartoon_Transition_Panel.svg
  - analysis/pca_mode_cartoon_overlay.pml
  - analysis/PCA_transition_*.pdb

python3 ../../Analysis/16_FEL/pca_mode_porcipine_panel.py \
  --projection analysis/PCA_projection.dat \
  --topology complex.parm7 \
  --trajectory cMD-Prod.nc \
  --outdir analysis/plots \
  --start-frame 1 \
  --end-frame 20000 \
  --transition-steps 7 \
  --pymol-bin pymol \
  --dpi 600


"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import List

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams


def set_style() -> None:
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = ["Cambria", "Times New Roman", "DejaVu Serif"]
    rcParams["mathtext.fontset"] = "stix"
    rcParams["font.size"] = 10.5
    rcParams["axes.linewidth"] = 1.0
    rcParams["xtick.direction"] = "in"
    rcParams["ytick.direction"] = "in"
    rcParams["xtick.top"] = False
    rcParams["ytick.right"] = False


def load_projection(path: Path) -> np.ndarray:
    arr = np.loadtxt(path, comments=["#", "@"]) 
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 2:
        raise SystemExit(f"Need at least frame + PC1 in {path}, got shape={arr.shape}")
    return arr


def explained_variance_from_projection(pcs: np.ndarray) -> np.ndarray:
    var = np.var(pcs, axis=0, ddof=1)
    total = float(np.sum(var))
    if total <= 0:
        return np.zeros_like(var)
    return var / total


def extract_frame_pdb(top: Path, traj: Path, frame: int, out_pdb: Path) -> None:
    cpptraj_in = f"""parm {top}
trajin {traj} {frame} {frame} 1
autoimage
trajout {out_pdb} pdb
run
"""
    proc = subprocess.run(
        ["cpptraj"],
        input=cpptraj_in,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(
            "cpptraj failed when extracting frame.\n"
            f"frame={frame} out={out_pdb}\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
    if not out_pdb.exists():
        raise SystemExit(f"Failed to create PDB: {out_pdb}")


def _hex_to_rgb01(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def _rgb01_to_hex(rgb: tuple[float, float, float]) -> str:
    return "#%02x%02x%02x" % tuple(max(0, min(255, int(round(c * 255)))) for c in rgb)


def gradient_hex(c0: str, c1: str, n: int) -> List[str]:
    a = np.array(_hex_to_rgb01(c0), dtype=float)
    b = np.array(_hex_to_rgb01(c1), dtype=float)
    if n <= 1:
        return [c0]
    cols = []
    for i in range(n):
        t = i / float(n - 1)
        c = (1.0 - t) * a + t * b
        cols.append(_rgb01_to_hex(tuple(c.tolist())))
    return cols


def build_pymol_script(
    pdb_paths: List[Path],
    pml_path: Path,
    out_png: Path,
    width: int,
    height: int,
    dpi: int,
) -> None:
    # Blue -> orange transition.
    colors = gradient_hex("#2f65d9", "#c07a1a", len(pdb_paths))

    lines: List[str] = []
    lines.append("reinitialize")
    lines.append("bg_color white")
    lines.append("set antialias, 2")
    lines.append("set orthoscopic, on")
    lines.append("set depth_cue, 0")
    lines.append("set ray_opaque_background, on")
    lines.append("set cartoon_fancy_helices, 1")
    lines.append("set cartoon_smooth_loops, 1")
    lines.append("set cartoon_sampling, 12")
    lines.append("set cartoon_side_chain_helper, on")
    lines.append("hide everything, all")

    for i, p in enumerate(pdb_paths):
        obj = f"m{i:02d}"
        cname = f"grad{i:02d}"
        lines.append(f"load {p.as_posix()}, {obj}")
        # Keep only protein to avoid solvent/ion cloud artifacts.
        lines.append(f"remove {obj} and not polymer.protein")
        lines.append(f"hide everything, {obj}")
        lines.append(f"set_color {cname}, [{_hex_to_rgb01(colors[i])[0]:.4f}, {_hex_to_rgb01(colors[i])[1]:.4f}, {_hex_to_rgb01(colors[i])[2]:.4f}]")
        lines.append(f"color {cname}, {obj}")
        lines.append(f"show cartoon, {obj}")

    # Align all frames to first by CA for clean overlay.
    for i in range(1, len(pdb_paths)):
        lines.append(f"align m{i:02d} and name CA, m00 and name CA")

    # Transparency gradient: early frames more transparent, endpoints stronger.
    n = len(pdb_paths)
    for i in range(n):
        obj = f"m{i:02d}"
        if i == 0 or i == n - 1:
            tr = 0.10
        else:
            tr = 0.60 - 0.45 * (i / float(max(1, n - 1)))
            tr = max(0.15, min(0.65, tr))
        lines.append(f"set cartoon_transparency, {tr:.3f}, {obj}")

    lines.append("orient m00")
    lines.append("zoom all, 1.2")
    lines.append(f"ray {int(width)}, {int(height)}")
    lines.append(f"png {out_png.as_posix()}, dpi={int(dpi)}")
    lines.append("quit")

    pml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_pymol_overlay(
    pdb_paths: List[Path],
    outdir: Path,
    pymol_bin: str,
    width: int,
    height: int,
    dpi: int,
) -> Path:
    if shutil.which(pymol_bin) is None:
        raise SystemExit(
            f"PyMOL executable not found: {pymol_bin}. "
            "Install PyMOL or pass --pymol-bin with a valid path."
        )
    pml = outdir.parent / "pca_mode_cartoon_overlay.pml"
    out_png = outdir / "16_PCA_Mode_Cartoon_Transition.png"
    build_pymol_script(pdb_paths, pml, out_png, width=width, height=height, dpi=dpi)

    proc = subprocess.run([pymol_bin, "-cq", str(pml)], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise SystemExit(
            "PyMOL rendering failed.\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
    if not out_png.exists():
        raise SystemExit(f"PyMOL did not produce output image: {out_png}")
    return out_png


def make_panel_from_png(var_ratio: np.ndarray, top_png: Path, out_svg: Path, top_pc: int, dpi: int) -> None:
    set_style()
    fig = plt.figure(figsize=(8.8, 7.6), facecolor="white", constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.35, 1.0])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])

    img = plt.imread(top_png)
    ax1.imshow(img)
    ax1.set_title("PCA Mode Visualization (Cartoon transition: 0 ns → 200 ns)")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_frame_on(False)

    n_pc = min(int(top_pc), int(var_ratio.size))
    idx = np.arange(1, n_pc + 1)
    pct = var_ratio[:n_pc] * 100.0
    cum = np.cumsum(pct)
    ax2.bar(idx, pct, color="#4c78a8", alpha=0.9, width=0.72, edgecolor="black", linewidth=0.4)
    ax2.plot(idx, cum, color="#e45756", marker="o", linewidth=1.7, markersize=4)
    ax2.set_xticks(idx)
    ax2.set_xticklabels([f"PC{i}" for i in idx])
    ax2.set_xlim(0.4, n_pc + 0.6)
    ax2.set_ylabel("Explained Variance (%)")
    ax2.set_xlabel("Principal Components")
    ax2.set_title("Eigenvalue / Variance Contribution")
    ax2.grid(False)

    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_svg, format="svg", bbox_inches="tight", dpi=int(dpi))
    plt.close(fig)


def make_top_svg_from_png(top_png: Path, out_svg: Path, dpi: int) -> None:
    """Wrap rendered PNG into an SVG figure file for manuscript pipeline consistency."""
    set_style()
    fig, ax = plt.subplots(figsize=(7.2, 5.4), facecolor="white", constrained_layout=True)
    img = plt.imread(top_png)
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    ax.set_title("PCA Mode Cartoon Transition (0 ns \u2192 200 ns)")
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_svg, format="svg", bbox_inches="tight", dpi=int(dpi))
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate PCA mode panel with PyMOL cartoon overlay + variance plot.")
    ap.add_argument("--projection", type=Path, default=Path("analysis/PCA_projection.dat"))
    ap.add_argument("--topology", type=Path, default=Path("complex.parm7"))
    ap.add_argument("--trajectory", type=Path, default=Path("cMD-Prod.nc"))
    ap.add_argument("--outdir", type=Path, default=Path("analysis/plots"))
    ap.add_argument("--top-pc", type=int, default=10)
    ap.add_argument("--transition-steps", type=int, default=10)
    ap.add_argument("--start-frame", type=int, default=1)
    ap.add_argument("--end-frame", type=int, default=20000)
    ap.add_argument("--pymol-bin", type=str, default="pymol")
    ap.add_argument("--width", type=int, default=2200)
    ap.add_argument("--height", type=int, default=1600)
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    if not args.projection.exists():
        raise SystemExit(f"Missing projection file: {args.projection}")
    if not args.topology.exists():
        raise SystemExit(f"Missing topology file: {args.topology}")
    if not args.trajectory.exists():
        raise SystemExit(f"Missing trajectory file: {args.trajectory}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    arr = load_projection(args.projection)
    pcs = arr[:, 1:]

    n_steps = max(2, int(args.transition_steps))
    frame_list = np.linspace(int(args.start_frame), int(args.end_frame), n_steps)
    frame_list = [int(round(x)) for x in frame_list]
    frame_list[0] = int(args.start_frame)
    frame_list[-1] = int(args.end_frame)

    tmp_dir = args.outdir.parent
    pdb_paths: List[Path] = []
    for i, fr in enumerate(frame_list):
        p = tmp_dir / f"PCA_transition_{i:02d}_f{fr}.pdb"
        extract_frame_pdb(args.topology, args.trajectory, fr, p)
        pdb_paths.append(p)

    top_png = render_pymol_overlay(
        pdb_paths=pdb_paths,
        outdir=args.outdir,
        pymol_bin=args.pymol_bin,
        width=args.width,
        height=args.height,
        dpi=args.dpi,
    )

    var_ratio = explained_variance_from_projection(pcs)
    top_svg = args.outdir / "16_PCA_Mode_Cartoon_Transition.svg"
    make_top_svg_from_png(top_png, top_svg, dpi=args.dpi)
    out_svg = args.outdir / "16_PCA_Mode_Cartoon_Transition_Panel.svg"
    make_panel_from_png(var_ratio, top_png, out_svg, top_pc=args.top_pc, dpi=args.dpi)

    print("[OK] PCA cartoon-transition panel generated")
    print(f"  - transition frames: {frame_list[0]} -> {frame_list[-1]} (steps={n_steps})")
    print(f"  - top image: {top_png.resolve()}")
    print(f"  - top svg: {top_svg.resolve()}")
    print(f"  - panel svg: {out_svg.resolve()}")
    print(f"  - pymol script: {(tmp_dir / 'pca_mode_cartoon_overlay.pml').resolve()}")


if __name__ == "__main__":
    main()
