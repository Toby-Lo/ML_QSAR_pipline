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

# Total H-bonds including solvent contributions.
hbond All out analysis/HBond_all.hbvtime.dat solventdonor :WAT solventacceptor :WAT@O \
  avgout analysis/HBond_all.UU.avg.dat solvout analysis/HBond_all.UV.avg.dat bridgeout analysis/HBond_all.bridge.avg.dat
run
