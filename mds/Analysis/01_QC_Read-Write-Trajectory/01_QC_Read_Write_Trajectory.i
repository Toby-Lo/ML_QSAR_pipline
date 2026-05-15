# === Usage (run from a simulation directory) ===
#   cd ./runs/18198196/
#   mkdir -p analysis
#   cpptraj -i "../../Analysis/01_QC_Read-Write-Trajectory/01_QC_Read_Write_Trajectory.i"
#
# === What to edit (common) ===
# - Topology: `parm complex.parm7` (only change if you renamed it)
# - Trajectory: `trajin cMD-Prod*.nc ...` (use your real trajectory; centered/imaged is recommended)
# - Masks: ligand residue name/number (e.g., `:UNL`), protein range (e.g., `:1-XXX`)
# - Sampling: `trajin ... 1 last 10` controls stride (here: every 10th frame)
# ==============================================
# plot:
# python3 ../../Analysis/01_QC_Read-Write-Trajectory/plot_timeseries.py analysis/QC_Calpha_RMSD.dat -o analysis/plots/01_QC_Calpha_RMSD_timeseries.svg

parm complex.parm7
trajin cMD-Prod.nc 1 last 10

# Basic sanity checks:
# - Images molecules back into a familiar unit cell so structural metrics don't explode.
autoimage

# - Writes a small stride trajectory to prove reading/writing works.
trajout analysis/QC_stride10.nc netcdf

# - Writes the first frame as PDB for quick visual inspection.
trajout analysis/QC_firstframe.pdb pdb onlyframes 1

# - Quick metric (Cα RMSD vs first frame).
rms first @CA out analysis/QC_Calpha_RMSD.dat
