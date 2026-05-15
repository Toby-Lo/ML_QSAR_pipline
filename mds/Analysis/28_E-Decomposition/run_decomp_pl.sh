#!/usr/bin/env bash
set -euo pipefail

# grep -nE "ligand_mask|receptor_mask" _MMPBSA_info FINAL_GBSA.dat

# Run from a simulation directory, e.g.:
#   cd mds/runs/a1a0m
#   bash ../../Analysis/28_E-Decomposition/run_decomp_pl.sh

MASK_FILE="${MASK_FILE:-analysis_masks.txt}"
DECOMP_TEMPLATE="${DECOMP_TEMPLATE:-../../Analysis/28_E-Decomposition/Decomposition.i}"
DECOMP_IN="${DECOMP_IN:-analysis/Decomposition.auto.in}"
OUT_PREFIX="${OUT_PREFIX:-analysis/DECOMP}"
PRINT_RES="${PRINT_RES:-AUTO}"
STARTFRAME="${STARTFRAME:-1}"
ENDFRAME="${ENDFRAME:-200}"
INTERVAL="${INTERVAL:-2}"
FORCE_ANTE="${FORCE_ANTE:-0}"

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
if [[ ! -f "$DECOMP_TEMPLATE" ]]; then
  echo "[ERROR] Missing template: $DECOMP_TEMPLATE" >&2
  exit 2
fi

RECEPTOR="$(awk -F= '/^[[:space:]]*RECEPTOR[[:space:]]*=/{gsub(/[[:space:]]/,"",$2); print $2; exit}' "$MASK_FILE")"
LIGAND="$(awk -F= '/^[[:space:]]*LIGAND[[:space:]]*=/{gsub(/[[:space:]]/,"",$2); print $2; exit}' "$MASK_FILE")"
if [[ -z "$RECEPTOR" || -z "$LIGAND" ]]; then
  echo "[ERROR] Failed to parse RECEPTOR/LIGAND from $MASK_FILE" >&2
  exit 3
fi

# Auto print_res: ligand + full receptor range.
if [[ "$PRINT_RES" == "AUTO" ]]; then
  # MMPBSA print_res expects integer selections/ranges (no ':' prefix).
  LIGAND_RES="${LIGAND#:}"
  RECEPTOR_RES="${RECEPTOR#:}"
  PRINT_RES="${LIGAND_RES};${RECEPTOR_RES}"
fi

mkdir -p analysis

# Render decomposition input from template.
awk -v lig="$LIGAND" -v rec="$RECEPTOR" -v pr="$PRINT_RES" \
    -v sf="$STARTFRAME" -v ef="$ENDFRAME" -v itv="$INTERVAL" '
{
  gsub(/__LIGAND_MASK__/, lig)
  gsub(/__RECEPTOR_MASK__/, rec)
  gsub(/__PRINT_RES__/, pr)
  if ($0 ~ /startframe=/) {
    gsub(/startframe=[0-9]+/, "startframe=" sf)
    gsub(/endframe=[0-9]+/, "endframe=" ef)
    gsub(/interval=[0-9]+/, "interval=" itv)
  }
  print
}' "$DECOMP_TEMPLATE" > "$DECOMP_IN"

echo "[INFO] RECEPTOR mask: $RECEPTOR"
echo "[INFO] LIGAND mask  : $LIGAND"
echo "[INFO] print_res    : $PRINT_RES"
echo "[INFO] input file   : $DECOMP_IN"

# Build stripped topologies needed by MMPBSA decomposition.
if [[ "$FORCE_ANTE" == "1" ]]; then
  rm -f complex_No_WAT.parm7 protein.parm7 ligand.parm7
fi

if [[ -f complex_No_WAT.parm7 && -f protein.parm7 && -f ligand.parm7 ]]; then
  echo "[INFO] Reusing existing topologies: complex_No_WAT.parm7, protein.parm7, ligand.parm7"
else
  ante-MMPBSA.py -p complex.parm7 -c complex_No_WAT.parm7 -r protein.parm7 -l ligand.parm7 -s ":WAT,NA,CL" -m "$RECEPTOR" --radii=mbondi3
fi

# Per-residue decomposition (GB/PB according to input file sections).
MMPBSA.py -O \
  -i "$DECOMP_IN" \
  -o "${OUT_PREFIX}_FINAL_RESULTS.dat" \
  -do "${OUT_PREFIX}_FINAL_DECOMP.dat" \
  -sp complex.parm7 \
  -cp complex_No_WAT.parm7 \
  -rp protein.parm7 \
  -lp ligand.parm7 \
  -y cMD-Prod.nc

echo "[OK] Decomposition complete"
echo "  - ${OUT_PREFIX}_FINAL_RESULTS.dat"
echo "  - ${OUT_PREFIX}_FINAL_DECOMP.dat"
