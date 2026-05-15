#!/usr/bin/env bash
set -euo pipefail

# grep -nE "ligand_mask|receptor_mask" _MMPBSA_info FINAL_GBSA.dat

# One-click PCA (step14) + FEL (step16)
# Run from simulation directory, e.g.:
#   cd mds/runs/a1a0m
#   bash ../../Analysis/16_FEL/run_pca_fel.sh

PCA_INPUT="${PCA_INPUT:-analysis/PCA.auto.i}"
PROJ_FILE="${PROJ_FILE:-analysis/PCA_projection.dat}"
FEL_INPUT="${FEL_INPUT:-analysis/FEL_input.xvg}"
FEL_XPM="${FEL_XPM:-analysis/Free-Energy-Landscape.xpm}"
FEL_TXT="${FEL_TXT:-analysis/Free-Energy-Landscape.txt}"
XPM2TXT="${XPM2TXT:-../../Analysis/16_FEL/xpm2txt.py}"
GMX_BIN="${GMX_BIN:-gmx}"
PLOT_SCRIPT="${PLOT_SCRIPT:-../../Analysis/16_FEL/pca_fel_publication_plots.py}"
MODE_PANEL_SCRIPT="${MODE_PANEL_SCRIPT:-../../Analysis/16_FEL/pca_mode_porcipine_panel.py}"
CARTOON_OVERLAY_SCRIPT="${CARTOON_OVERLAY_SCRIPT:-../../Analysis/16_FEL/make_pca_cartoon_overlay.sh}"

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[ERROR] Missing command: $1" >&2
    exit 2
  fi
}

need_cmd cpptraj
need_cmd python3

mkdir -p analysis

# Build a robust two-stage PCA cpptraj input (avoids old template ordering issues).
cat > "$PCA_INPUT" <<'CPPTRAJ_EOF'
parm complex.parm7
trajin cMD-Prod.nc 1 last 10
autoimage
rms first @CA
matrix covar name COVAR @CA
run

runanalysis diagmatrix COVAR out analysis/PCA_evecs.dat vecs 10 name EVECS

clear trajin
trajin cMD-Prod.nc 1 last 10
autoimage
rms first @CA
projection PC modes EVECS out analysis/PCA_projection.dat beg 1 end 3 @CA
hist PC:1 bins 100 out analysis/PCA_hist.agr norm name PC1
hist PC:2 bins 100 out analysis/PCA_hist.agr norm name PC2
hist PC:3 bins 100 out analysis/PCA_hist.agr norm name PC3
run
CPPTRAJ_EOF

echo "[1/4] Running PCA via cpptraj ..."
cpptraj -i "$PCA_INPUT"

if [[ ! -f "$PROJ_FILE" ]]; then
  echo "[ERROR] Missing projection file: $PROJ_FILE" >&2
  exit 3
fi

echo "[2/4] Building FEL input from PC1/PC2 ..."
# Keep rows with at least 3 numeric columns: frame pc1 pc2 [...]
awk 'NF>=3 {print $1, $2, $3}' "$PROJ_FILE" > "$FEL_INPUT"

if [[ ! -s "$FEL_INPUT" ]]; then
  echo "[ERROR] FEL input is empty: $FEL_INPUT" >&2
  exit 3
fi

if command -v "$GMX_BIN" >/dev/null 2>&1; then
  echo "[3/5] Running gmx sham ..."
  "$GMX_BIN" sham -f "$FEL_INPUT" -ls "$FEL_XPM"
  if [[ ! -f "$FEL_XPM" ]]; then
    echo "[ERROR] FEL xpm not generated: $FEL_XPM" >&2
    exit 4
  fi

  echo "[4/5] Converting XPM -> txt ..."
  python3 "$XPM2TXT" -f "$FEL_XPM" -o "$FEL_TXT"
else
  echo "[3/5] gmx not found; skipping gmx sham/XPM conversion."
  echo "       (AMBER-only mode: FEL contour/surface will be generated directly from PCA projection.)"
fi

if [[ -f "$PLOT_SCRIPT" ]]; then
  echo "[5/5] Generating publication-style PCA+FEL figures ..."
  python3 "$PLOT_SCRIPT" \
    --projection "$PROJ_FILE" \
    --outdir analysis/plots
else
  echo "[WARN] Plot script not found: $PLOT_SCRIPT"
fi

if [[ -f "$MODE_PANEL_SCRIPT" ]]; then
  echo "[Extra] Generating PCA mode cartoon-transition panel ..."
  python3 "$MODE_PANEL_SCRIPT" \
    --projection "$PROJ_FILE" \
    --topology complex.parm7 \
    --trajectory cMD-Prod.nc \
    --outdir analysis/plots \
    --top-pc 10 \
    --start-frame 1 \
    --end-frame 20000 \
    --transition-steps 7 \
    --dpi 600 || true
else
  echo "[WARN] Mode panel script not found: $MODE_PANEL_SCRIPT"
fi

if [[ -f "$CARTOON_OVERLAY_SCRIPT" ]]; then
  if command -v "${PYMOL_BIN:-pymol}" >/dev/null 2>&1; then
    echo "[Extra] Generating PyMOL cartoon overlay (0 ns vs 200 ns) ..."
    bash "$CARTOON_OVERLAY_SCRIPT" || true
  else
    echo "[Extra] PyMOL not found; skip cartoon overlay."
  fi
else
  echo "[WARN] Cartoon overlay script not found: $CARTOON_OVERLAY_SCRIPT"
fi

echo "[OK] PCA + FEL complete"
echo "  - Projection : $PROJ_FILE"
echo "  - FEL input  : $FEL_INPUT"
if [[ -f "$FEL_XPM" ]]; then
  echo "  - FEL xpm    : $FEL_XPM"
fi
if [[ -f "$FEL_TXT" ]]; then
  echo "  - FEL text   : $FEL_TXT"
fi
