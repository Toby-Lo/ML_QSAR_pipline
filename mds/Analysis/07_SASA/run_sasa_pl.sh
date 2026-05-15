#!/usr/bin/env bash
set -euo pipefail

# grep -nE "ligand_mask|receptor_mask" _MMPBSA_info FINAL_GBSA.dat

# Run from a simulation directory, e.g.:
#   cd mds/runs/a1a0m
#   bash ../../Analysis/07_SASA/run_sasa_pl.sh
#
# Plotting:
# python3 ../../Analysis/07_SASA/plot_timeseries.py analysis/SASA_protein.dat -o analysis/plots/07_SASA_protein.svg
# python3 ../../Analysis/07_SASA/plot_timeseries.py analysis/SASA_ligand.dat  -o analysis/plots/07_SASA_ligand.svg

MASK_FILE="${MASK_FILE:-analysis_masks.txt}"
is_valid_mask() {
  local m="${1:-}"
  [[ -n "$m" && "$m" == :* && "$m" != "receptor_mask" && "$m" != "ligand_mask" ]]
}
bootstrap_masks() {
  if [[ -f "$MASK_FILE" ]]; then
    local r0 l0
    r0="$(awk -F= '/^[[:space:]]*RECEPTOR[[:space:]]*=/{gsub(/[[:space:]]/,"",$2); print $2; exit}' "$MASK_FILE")"
    l0="$(awk -F= '/^[[:space:]]*LIGAND[[:space:]]*=/{gsub(/[[:space:]]/,"",$2); print $2; exit}' "$MASK_FILE")"
    if is_valid_mask "$r0" && is_valid_mask "$l0"; then
      return 0
    fi
    echo "[WARN] Existing $MASK_FILE is invalid, regenerating from MMPBSA records..."
  fi
  local rec="" lig=""
  if [[ -f "_MMPBSA_info" ]]; then
    rec="$(sed -n "s/.*INPUT\\['receptor_mask'\\][[:space:]]*=[[:space:]]*'\\([^']*\\)'.*/\\1/p" _MMPBSA_info | head -n1)"
    lig="$(sed -n "s/.*INPUT\\['ligand_mask'\\][[:space:]]*=[[:space:]]*'\\([^']*\\)'.*/\\1/p" _MMPBSA_info | head -n1)"
  fi
  if [[ -z "$rec" || -z "$lig" ]] && [[ -f "FINAL_GBSA.dat" ]]; then
    rec="$(sed -n "s/.*receptor_mask='\\([^']*\\)'.*/\\1/p" FINAL_GBSA.dat | head -n1)"
    lig="$(sed -n "s/.*ligand_mask='\\([^']*\\)'.*/\\1/p" FINAL_GBSA.dat | head -n1)"
  fi
  if is_valid_mask "$rec" && is_valid_mask "$lig"; then
    printf "RECEPTOR=%s\nLIGAND=%s\n" "$rec" "$lig" > "$MASK_FILE"
    echo "[INFO] Auto-generated $MASK_FILE from MMPBSA records."
    return 0
  fi
  echo "[ERROR] Missing $MASK_FILE and could not infer masks from _MMPBSA_info/FINAL_GBSA.dat" >&2
  exit 2
}
bootstrap_masks

RECEPTOR="$(awk -F= '/^[[:space:]]*RECEPTOR[[:space:]]*=/{gsub(/[[:space:]]/,"",$2); print $2; exit}' "$MASK_FILE")"
LIGAND="$(awk -F= '/^[[:space:]]*LIGAND[[:space:]]*=/{gsub(/[[:space:]]/,"",$2); print $2; exit}' "$MASK_FILE")"

if [[ -z "$RECEPTOR" || -z "$LIGAND" ]]; then
  echo "[ERROR] Failed to parse RECEPTOR/LIGAND from $MASK_FILE" >&2
  exit 3
fi

mkdir -p analysis

echo "[INFO] RECEPTOR mask: $RECEPTOR"
echo "[INFO] LIGAND mask  : $LIGAND"

cpptraj <<CPPTRAJ_EOF
parm complex.parm7
trajin cMD-Prod.nc 1 last 10
autoimage

# Protein and ligand SASA split
surf ${RECEPTOR} out analysis/SASA_protein.dat
surf ${LIGAND} out analysis/SASA_ligand.dat

# Optional: total solute SASA for reference
surf out analysis/SASA_total.dat
run
CPPTRAJ_EOF

echo "[OK] SASA analysis complete"
echo "  - analysis/SASA_protein.dat"
echo "  - analysis/SASA_ligand.dat"
echo "  - analysis/SASA_total.dat"
