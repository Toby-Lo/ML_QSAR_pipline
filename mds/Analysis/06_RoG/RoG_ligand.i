# === Usage (run from a simulation directory) ===
#   Ligand RoG (heavy atoms by default)!
#   cd runs/<SYS>
#   mkdir -p analysis
#   cpptraj -i ../../Analysis/06_RoG/RoG_ligand.i
#
# === What to edit (common) ===
# - Topology: `parm complex.parm7` (only change if you renamed it)
# - Trajectory: `trajin cMD-Prod*.nc ...` (use your real trajectory; centered/imaged is recommended)
# - Ligand mask: default is `:UNL&!@H=` (edit to your ligand residue name/number)
# - Sampling: `trajin ... 1 last 10` controls stride (here: every 10th frame)
# ==============================================
# Plotting:
# python3 ../../Analysis/06_RoG/plot_timeseries.py analysis/RoG_Ligand.dat -o analysis/plots/06_RoG_Ligand.svg

parm complex.parm7
trajin cMD-Prod.nc 1 last 10
autoimage

# Radius of gyration for ligand (exclude hydrogens by default).
# NOTE: change `:UNL` if your ligand residue name differs.
radgyr :UNL&!@H= out analysis/RoG_Ligand.dat mass nomax
run

# Note: it is important to remove water from trajectories and topology.
