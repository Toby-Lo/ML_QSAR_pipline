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
trajin cMD-Prod.nc 1 last 10
autoimage

# EDIT ME: pick a residue range / atom selection you care about.
# Example:
#   2drms :280-310@C,N,O,CA,CB&!@H= rmsout analysis/2D_RMSD_280_310.gnu
#
# Optional RMSF for a selection:
#   atomicfluct out analysis/RMSF_selection.dat :280-310 byres
run
