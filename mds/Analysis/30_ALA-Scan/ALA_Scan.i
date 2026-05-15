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

sample input file for running alanine scanning
 &general
   startframe=150, endframe=200, interval=1, ligand_mask=:598-793, receptor_mask=:1-597,
/
&gb
  saltcon=0.1
/
&pb
  inp=1, istrng=0.100, radiopt=0,
/
&alanine_scanning
/
