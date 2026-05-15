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

Per-residue GB/PB decomposition (template for run_decomp_pl.sh).
The runner will replace:
  __LIGAND_MASK__
  __RECEPTOR_MASK__
  __PRINT_RES__

&general
startframe=1, endframe=200, interval=2, ligand_mask=__LIGAND_MASK__, receptor_mask=__RECEPTOR_MASK__, netcdf=1
/
&gb
igb=5, saltcon=0.150,
/
&pb
inp=1, istrng=0.15, radiopt=0,
/
&decomp
  idecomp=1, print_res="__PRINT_RES__"
  dec_verbose=1,
/
