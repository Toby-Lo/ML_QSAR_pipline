# === Usage (run from a simulation directory) ===
#    Cα-based RoG!
#   cd runs/<SYS>
#   mkdir -p analysis
#   cpptraj -i  ../../Analysis/06_RoG/RoG.i 
#
# === What to edit (common) ===
# - Topology: `parm complex.parm7` (only change if you renamed it)
# - Trajectory: `trajin cMD-Prod*.nc ...` (use your real trajectory; centered/imaged is recommended)
# - Masks: ligand residue name/number (e.g., `:UNL`), protein range (e.g., `:1-XXX`)
# - Sampling: `trajin ... 1 last 10` controls stride (here: every 10th frame)
# ==============================================
# Plotting:
# python3 ../../Analysis/06_RoG/plot_timeseries.py analysis/RoG_Calpha.dat -o analysis/plots/06_RoG_Calpha.svg


parm complex.parm7
trajin cMD-Prod.nc 1 last 10
autoimage

# Radius of gyration for protein Cα atoms (edit mask if needed).
radgyr @CA out analysis/RoG_Calpha.dat mass nomax
run

 
 
 #Note: it is important to remove water from trajectories and topology.
