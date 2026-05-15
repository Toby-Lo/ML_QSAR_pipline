# === Usage (recommended, run from a simulation directory) ===
#   cd runs/<SYS>
#   bash ../../Analysis/07_SASA/run_sasa_pl.sh
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

# Legacy fallback: total SASA.
# For publication-grade Protein/Ligand split SASA, use run_sasa_pl.sh + analysis_masks.txt.
surf out analysis/SASA_total.dat
run



#surf :1 out surf.dat
#calculate the overall surface area of all solute atoms, as well as the contribution of residue 1 to the overall surface area, writing both results to “surf.dat”:
