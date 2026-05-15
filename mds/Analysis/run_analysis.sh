#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage (run from a simulation directory, e.g. runs/18198196/):
  bash ../../Analysis/run_analysis.sh [START_STEP] [END_STEP]

Options:
  START_STEP / END_STEP are numeric prefixes like 01, 02, ..., 19

Outputs:
  Writes to ./analysis/ in the current working directory.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

START_STEP="${1:-01}"
END_STEP="${2:-99}"

ANALYSIS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$PWD"

if ! command -v cpptraj >/dev/null 2>&1; then
  echo "[error] cpptraj not found in PATH (activate your amber env first)." >&2
  exit 127
fi

mkdir -p "${RUN_DIR}/analysis"

step_in_range() {
  local step="$1"
  (( 10#$step >= 10#$START_STEP && 10#$step <= 10#$END_STEP ))
}

run_cpptraj() {
  local rel="$1"
  local step="${rel%%_*}"
  if step_in_range "$step"; then
    echo "[run] cpptraj -i ${rel}"
    cpptraj -i "${ANALYSIS_DIR}/${rel}"
  fi
}

run_cpptraj "01_QC_Read-Write-Trajectory/01_QC_Read_Write_Trajectory.i"
run_cpptraj "02_RMSD-RMSF/RMSD_RMSF.i"
run_cpptraj "03_Ligand-RMSD/LIG-RMSD.i"
run_cpptraj "04_RMSD-Histogram/RMSD-Hist.i"
run_cpptraj "05_DSSP/DSSP.i"
run_cpptraj "06_RoG/RoG.i"
run_cpptraj "07_SASA/SASA.i"
run_cpptraj "08_HBond-Total/H-Bond-Analysis.i"
run_cpptraj "09_HBond-Complex/H-Bond-Small-Moleculex-Complex.i"
run_cpptraj "12_2D-RMSD/2D-RMSD-All-Residues.i"
run_cpptraj "14_PCA/PCA.i"
run_cpptraj "17_DCCM/DCCM.i"
run_cpptraj "19_Average-Structure/Avg-PDB.i"

echo "[done] outputs: ${RUN_DIR}/analysis/"

