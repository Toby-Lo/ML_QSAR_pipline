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

parm complex.parm7
trajin Name_combine.nc
strip :WAT,NA,CL
trajout Name_frame1.pdb onlyframes 1 pdb
trajout Nameframe11000.pdb onlyframes 11000 pdb


# The Name should be replace with the Trajectory Name. 1 and 11000 are the frames numbers to be saved from the trajetor.
