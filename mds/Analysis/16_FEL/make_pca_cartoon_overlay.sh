#!/usr/bin/env bash
set -euo pipefail

# Generate PyMOL cartoon overlay figure for two trajectory endpoints (e.g., 0 ns vs 200 ns).
# Run from a simulation directory (mds/runs/<system>):
#   bash ../../Analysis/16_FEL/make_pca_cartoon_overlay.sh
#
# Optional envs:
#   TOP=complex.parm7
#   TRAJ=cMD-Prod.nc
#   FRAME_START=1
#   FRAME_END=20000
#   OUTDIR=analysis/plots
#   PYMOL_BIN=pymol

TOP="${TOP:-complex.parm7}"
TRAJ="${TRAJ:-cMD-Prod.nc}"
FRAME_START="${FRAME_START:-1}"
FRAME_END="${FRAME_END:-20000}"
OUTDIR="${OUTDIR:-analysis/plots}"
PYMOL_BIN="${PYMOL_BIN:-pymol}"

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[ERROR] Missing command: $1" >&2
    exit 2
  fi
}

need_cmd cpptraj
need_cmd "$PYMOL_BIN"

mkdir -p analysis "$OUTDIR"

PDB0="analysis/PCA_mode_0ns.pdb"
PDB1="analysis/PCA_mode_200ns.pdb"
PML="analysis/pca_mode_overlay.pml"
OUTPNG="$OUTDIR/16_PCA_Mode_Cartoon_0ns_200ns.png"

echo "[1/3] Extracting endpoint frames with cpptraj ..."
cpptraj <<EOF
parm $TOP
trajin $TRAJ $FRAME_START $FRAME_START 1
autoimage
trajout $PDB0 pdb
run

clear trajin
trajin $TRAJ $FRAME_END $FRAME_END 1
autoimage
trajout $PDB1 pdb
run
EOF

if [[ ! -f "$PDB0" || ! -f "$PDB1" ]]; then
  echo "[ERROR] Failed to generate endpoint PDBs." >&2
  exit 3
fi

echo "[2/3] Building PyMOL script ..."
cat > "$PML" <<'PML'
reinitialize
set ray_opaque_background, on
bg_color white
set antialias, 2
set orthoscopic, on
set depth_cue, 0
set cartoon_fancy_helices, 1
set cartoon_smooth_loops, 1
set cartoon_sampling, 14

load analysis/PCA_mode_0ns.pdb, state0
load analysis/PCA_mode_200ns.pdb, state1

remove solvent
hide everything, all
show cartoon, state0 or state1

color marine, state0
color tv_orange, state1

set cartoon_transparency, 0.12, state0
set cartoon_transparency, 0.12, state1

align state1 and name CA, state0 and name CA
orient state0 or state1
zoom state0 or state1, 1.2

# Small label anchors for time states
pseudoatom label0, pos=[0,0,0]
label label0, "0 ns"
set label_color, marine, label0
set label_size, 18, label0

pseudoatom label1, pos=[15,0,0]
label label1, "200 ns"
set label_color, tv_orange, label1
set label_size, 18, label1

set ray_trace_mode, 1
set ray_trace_gain, 0.08
set specular, 0.22
set shininess, 20
set cartoon_side_chain_helper, on

ray 2200,1600
png analysis/plots/16_PCA_Mode_Cartoon_0ns_200ns.png, dpi=600
quit
PML

echo "[3/3] Rendering with PyMOL ..."
"$PYMOL_BIN" -cq "$PML"

echo "[OK] PCA cartoon overlay generated:"
echo "  - $OUTPNG"

