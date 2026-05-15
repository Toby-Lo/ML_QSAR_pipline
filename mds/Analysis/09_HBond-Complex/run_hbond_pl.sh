#!/usr/bin/env bash
set -euo pipefail

# grep -nE "ligand_mask|receptor_mask" _MMPBSA_info FINAL_GBSA.dat

# Run from a simulation directory, e.g.:
#   cd mds/runs/a1a0m
#   bash ../../Analysis/09_HBond-Complex/run_hbond_pl.sh
# Plotting:
# python3 ../../Analysis/09_HBond-Complex/plot_timeseries.py analysis/HBond_PL.gnu -o analysis/plots/09_HBond_PL_TimeSeries.svg
# python3 ../../Analysis/09_HBond-Complex/plot_timeseries.py analysis/HBond_PL_p2l.hbvtime.dat -o analysis/plots/09_HBond_PL_p2l_TimeSeries.svg

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

# Parse KEY=VALUE lines, ignore comments/empty lines.
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

hbond PL_p2l out analysis/HBond_PL_p2l.hbvtime.dat \\
  donormask ${RECEPTOR} acceptormask ${LIGAND} \\
  avgout analysis/HBond_PL_p2l.avg.dat series uuseries analysis/HBond_PL_p2l.gnu nointramol

hbond PL_l2p out analysis/HBond_PL_l2p.hbvtime.dat \\
  donormask ${LIGAND} acceptormask ${RECEPTOR} \\
  avgout analysis/HBond_PL_l2p.avg.dat series uuseries analysis/HBond_PL_l2p.gnu nointramol

run

lifetime PL_p2l[solutehb] out analysis/HBond_PL_p2l_lifetime.dat
lifetime PL_l2p[solutehb] out analysis/HBond_PL_l2p_lifetime.dat
runanalysis
CPPTRAJ_EOF

echo "[OK] HBond protein-ligand analysis complete"
echo "  - analysis/HBond_PL_p2l.gnu"
echo "  - analysis/HBond_PL_l2p.gnu"
echo "  - analysis/HBond_PL_p2l.avg.dat"
echo "  - analysis/HBond_PL_l2p.avg.dat"
echo "  - analysis/HBond_PL_p2l_lifetime.dat"
echo "  - analysis/HBond_PL_l2p_lifetime.dat"
