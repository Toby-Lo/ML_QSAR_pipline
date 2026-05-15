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
trajin Name_combine.nc 1 10000 1000
autoimage
rms fit :1-306
strip :WAT
strip :WAT,NA,CL
trajout Name_Ensembal.pdb pdb
run 


#1000 is the offset of the trajectory read. start is 1 upto 10000 with 1000 gape total 10 stuctures generated.
