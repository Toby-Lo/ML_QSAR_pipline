#!/usr/bin/env bash
set -euo pipefail

# One-click batch analysis for all systems under mds/runs/*
# Usage:
#   bash mds/run_all_system_analyses.sh
# Optional:
#   RUNS_ROOT=mds/runs bash mds/run_all_system_analyses.sh
#   MISSING_ONLY=1 bash mds/run_all_system_analyses.sh
#   MODE=check   bash mds/run_all_system_analyses.sh  # print missing only
#   MODE=missing bash mds/run_all_system_analyses.sh  # run only missing
#   MODE=all     bash mds/run_all_system_analyses.sh  # run all

# for d in mds/runs/*; do [ -d "$d" ] || continue; (cd "$d" && bash ../../Analysis/16_FEL/run_pca_fel.sh); done
# for d in mds/runs/*; do [ -d "$d" ] || continue; [ -f "$d/analysis/plots/16_PCA_Mode_Cartoon_Transition_Panel.svg" ] && continue; (cd "$d" && bash ../../Analysis/16_FEL/run_pca_fel.sh); done


ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RUNS_ROOT="${RUNS_ROOT:-$ROOT_DIR/mds/runs}"
MISSING_ONLY="${MISSING_ONLY:-0}"
MODE="${MODE:-}"

# Backward compatibility: MISSING_ONLY=1 implies MODE=missing unless MODE already set.
if [[ -z "$MODE" ]]; then
  if [[ "$MISSING_ONLY" == "1" ]]; then
    MODE="missing"
  else
    MODE="all"
  fi
fi

case "$MODE" in
  all|missing|check) ;;
  *)
    echo "[ERROR] Invalid MODE=$MODE (allowed: all|missing|check)" >&2
    exit 1
    ;;
esac

if [[ "$MODE" == "check" ]]; then
  python3 "$ROOT_DIR/mds/aggregate_md_results.py" check --runs-root "$RUNS_ROOT"
  exit 0
fi

need_any_missing() {
  local run_dir="$1"
  shift
  for rel in "$@"; do
    if [[ ! -f "$run_dir/$rel" ]]; then
      return 0
    fi
  done
  return 1
}

should_run() {
  local run_dir="$1"
  shift
  if [[ "$MODE" == "all" ]]; then
    return 0
  fi
  need_any_missing "$run_dir" "$@"
}

for run in "$RUNS_ROOT"/*; do
  [[ -d "$run" ]] || continue
  sys="$(basename "$run")"
  echo "=================================================="
  echo "[SYSTEM] $sys"
  echo "=================================================="

  pushd "$run" >/dev/null
  mkdir -p analysis/plots

  # 01 QC
  if should_run "$run" "analysis/QC_Calpha_RMSD.dat"; then
    cpptraj -i ../../Analysis/01_QC_Read-Write-Trajectory/01_QC_Read_Write_Trajectory.i || true
    python3 ../../Analysis/01_QC_Read-Write-Trajectory/plot_timeseries.py analysis/QC_Calpha_RMSD.dat -o analysis/plots/01_QC_Calpha_RMSD_timeseries.svg || true
  fi

  # 02 RMSD/RMSF
  if should_run "$run" "analysis/Calpha_RMSD.dat" "analysis/RMSF.dat"; then
    cpptraj -i ../../Analysis/02_RMSD-RMSF/RMSD_RMSF.i || true
    python3 ../../Analysis/02_RMSD-RMSF/plot_timeseries.py analysis/Calpha_RMSD.dat -o analysis/plots/02_Calpha_RMSD.svg || true
    python3 ../../Analysis/02_RMSD-RMSF/plot_timeseries.py analysis/RMSF.dat -o analysis/plots/02_RMSF_Profile.svg || true
  fi

  # 03 Ligand RMSD
  if should_run "$run" "analysis/Ligand_RMSD.dat"; then
    cpptraj -i ../../Analysis/03_Ligand-RMSD/LIG-RMSD.i || true
    python3 ../../Analysis/03_Ligand-RMSD/plot_timeseries.py analysis/Ligand_RMSD.dat -o analysis/plots/03_Ligand_RMSD.svg || true
  fi

  # 06 RoG
  if should_run "$run" "analysis/RoG_Calpha.dat"; then
    cpptraj -i ../../Analysis/06_RoG/RoG.i || true
    python3 ../../Analysis/06_RoG/plot_timeseries.py analysis/RoG_Calpha.dat -o analysis/plots/06_RoG_Calpha.svg || true
  fi

  # 07 SASA (auto mask bootstrap)
  if should_run "$run" "analysis/SASA_protein.dat" "analysis/SASA_ligand.dat"; then
    bash ../../Analysis/07_SASA/run_sasa_pl.sh || true
    python3 ../../Analysis/07_SASA/plot_timeseries.py analysis/SASA_protein.dat -o analysis/plots/07_SASA_protein.svg || true
    python3 ../../Analysis/07_SASA/plot_timeseries.py analysis/SASA_ligand.dat -o analysis/plots/07_SASA_ligand.svg || true
  fi

  # 09 HBond + occupancy
  if should_run "$run" "analysis/HBond_PL_p2l.hbvtime.dat" "analysis/HBond_PL_l2p.hbvtime.dat"; then
    bash ../../Analysis/09_HBond-Complex/run_hbond_pl.sh || true
    python3 ../../Analysis/09_HBond-Complex/plot_timeseries.py analysis/HBond_PL_p2l.hbvtime.dat -o analysis/plots/09_HBond_PL_p2l_TimeSeries.svg || true
  fi
  if should_run "$run" "analysis/HBond_PL_occupancy_summary.csv"; then
    python3 ../../Analysis/09_HBond-Complex/hbond_occupancy_report.py \
      --p2l analysis/HBond_PL_p2l.avg.dat \
      --l2p analysis/HBond_PL_l2p.avg.dat \
      --out-csv analysis/HBond_PL_occupancy_summary.csv \
      --out-fig analysis/plots/HBond_PL_occupancy_top15.svg \
      --topn 15 || true
  fi

  # 16 PCA + FEL publication plots (AMBER-only mode supported)
  if should_run "$run" "analysis/PCA_projection.dat" "analysis/plots/16_FEL_PC1_PC2_contour.svg"; then
    bash ../../Analysis/16_FEL/run_pca_fel.sh || true
  fi

  # 27 MMGBSA
  if should_run "$run" \
    "MMGBSA_vs_time.dat" \
    "MMGBSA_vs_time_last50ns.dat" \
    "MMGBSA_summary.csv" \
    "MMGBSA_summary_last50ns.csv" \
    "FINAL_GBSA.dat"; then
    bash ../../Analysis/27_MMPBSA-GBSA/GBSA-vs-Time.i || true
  fi
  if should_run "$run" "analysis/plots/27_MMGBSA_vs_time.svg"; then
    python3 ../../Analysis/27_MMPBSA-GBSA/plot_mmgbsa_vs_time.py MMGBSA_vs_time.dat -o analysis/plots/27_MMGBSA_vs_time.svg --window 25 || true
  fi

  # Zn coordination distances
  if should_run "$run" "analysis/ZN221_CYM161_ZN_SG.dat" "analysis/plots/00_Zn_Coordination_Stability.svg"; then
    cpptraj -i ../../Analysis/zn_distances/zn_distances.cpptraj || true
    python3 ../../Analysis/zn_distances/plot_zn_distances.py || true
  fi

  popd >/dev/null

done

echo "[DONE] Batch analysis finished for systems under: $RUNS_ROOT (MODE=$MODE)"
