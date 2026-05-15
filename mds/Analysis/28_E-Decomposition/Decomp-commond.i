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

# Recommended:
#   bash ../../Analysis/28_E-Decomposition/run_decomp_pl.sh
# which auto-reads analysis_masks.txt and avoids hard-coded masks.

ante-MMPBSA.py -p complex.parm7 -c complex_No_WAT.parm7 -r protein.parm7 -l ligand.parm7 -s ":WAT,NA,CL" -m ':1-140' --radii=mbondi2




MMPBSA.py.MPI -O -i Decomposition.i -o Name-FINAL_RESULTS_MMPBSA.dat -do Name-FINAL_DECOMP_MMPBSA.dat -sp complex.parm7 -cp complex_No_WAT.parm7 -rp protein.parm7 -lp ligand.parm7 -y Name_combine.nc







export DO_PARALLEL='mpirun --mca btl ^openib --allow-run-as-root --use-hwthread-cpus -np 20'
