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
check @C,N,O,CA,CB skipbadframes silent
#
# EDIT ME: choose residue ranges you care about.
multidihedral chainone phi psi resrange 10-12 out analysis/dihedral-chain-one.dat
multidihedral chaintwo phi psi resrange 31-33 out analysis/dihedral-chain-two.dat
multidihedral chainthree phi psi resrange 52-54 out analysis/dihedral-chain-three.dat
run
multihist chainone[*] out analysis/dihedral-chain-one.hist normint min -180 max 180 step 1
multihist chaintwo[*] out analysis/dihedral-chain-two.hist normint min -180 max 180 step 1
multihist chainthree[*] out analysis/dihedral-chain-three.hist normint min -180 max 180 step 1
run
