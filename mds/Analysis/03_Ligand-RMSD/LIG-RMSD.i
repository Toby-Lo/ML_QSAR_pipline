# === Usage (run from a simulation directory) ===
#   cd runs/<SYS>
#   mkdir -p analysis
#   cpptraj -i ../../Analysis/<STEP>/<FILE>.i
#
# === What to edit (common) ===
# - Topology: `parm complex.parm7` (only change if you renamed it)
# - Trajectory: `trajin cMD-Prod*.nc ...` (use your real trajectory; centered/imaged is recommended)
# - Masks: ligand residue name/number (e.g., `:UNL`), protein range (e.g., `:1-XXX`)
# - Sampling: `trajin ... 1 last 10` controls stride (here: every 10th frame)
# ==============================================
# python3  ../../Analysis/03_Ligand-RMSD/plot_timeseries.py analysis/Ligand_RMSD.dat -o analysis/plots/03_Ligand_RMSD.svg

# mkdir -p analysis
# cpptraj -i ../../Analysis/03_Ligand-RMSD/LIG-RMSD.i

parm complex.parm7
trajin cMD-Prod.nc 1 last 10

autoimage

rms first @CA

rms first :UNL&!@H= out analysis/Ligand_RMSD.dat nofit
