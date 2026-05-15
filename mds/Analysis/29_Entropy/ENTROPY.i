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

Input file for running entropy calculations using NMode
&general
   startframe=18001, endframe=21000, interval=2, ligand_mask=:195, receptor_mask=:1-194, keep_files=1, netcdf=1, verbose=2,
/
&nmode
   nmstartframe=1, nmendframe=1500,
   nminterval=5, nmode_igb=1, nmode_istrng=0.1,
/
