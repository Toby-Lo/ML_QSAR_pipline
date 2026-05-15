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

# NOTE:
#   IRED setup is system-specific (you must define the correct N–H vectors).
#   This file is intentionally a stub template to avoid invalid cpptraj syntax.
#
# Example skeleton (edit atom selections!):
#   vector v1 :10@N ired :10@H
#   vector v2 :11@N ired :11@H
#   matrix ired name matired order 2
#   diagmatrix matired vecs 6 out analysis/ired.vec name ired_vec
#   ired relax NHdist 1.02 freq 500.0 tstep 1.0 tcorr 100.0 out analysis/ired.out noe order 2

run
