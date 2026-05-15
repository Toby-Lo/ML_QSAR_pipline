# === Usage (run from a simulation directory) ===
#   cd runs/<SYS>
#   mkdir -p analysis
#   cpptraj -i ../../Analysis/24_Ligand-Position/ligand-position.i
#
# === What to edit (common) ===
# - Topology: `parm complex.parm7` (only change if you renamed it)
# - Trajectory: `trajin cMD-Prod*.nc ...` (use your real trajectory; centered/imaged is recommended)
# - Masks: ligand residue name/number (e.g., `:UNL`), protein range (e.g., `:1-XXX`)
# - Sampling: `trajin ... 1 last 10` controls stride (here: every 10th frame)
# ==============================================

parm complex.parm7
trajin cMD-Prod_centered.nc 1 last 10
strip :WAT,Cl-,Na+

autoimage

rms fit :170-181&!@H= mass
distance lig_com :UNL :170-181 out analysis/lig_com_dist.dat
bounds :170-181|:UNL dx 0.5 name MyGrid out analysis/boundaries.dat

average average.pdb pdb

createcrd MyCoords
run

crdaction MyCoords grid analysis/ligand-grid.xplor data MyGrid :UNL
go
