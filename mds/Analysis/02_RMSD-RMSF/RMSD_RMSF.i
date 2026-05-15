# === Usage (run from a simulation directory) ===
#   cd runs/<SYS>
#   mkdir -p analysis
#   cpptraj -i ../../Analysis/02_RMSD-RMSF/RMSD_RMSF.i
#
# === What to edit (common) ===
# - Topology: `parm complex.parm7` (only change if you renamed it)
# - Trajectory: `trajin cMD-Prod*.nc ...` (use your real trajectory; centered/imaged is recommended)
# - Masks: ligand residue name/number (e.g., `:UNL`), protein range (e.g., `:1-XXX`)
# - Sampling: `trajin ... 1 last 10` controls stride (here: every 10th frame)
# ==============================================
# python3 ../../Analysis/02_RMSD-RMSF/plot_timeseries.py analysis/Calpha_RMSD.dat -o analysis/plots/02_Calpha_RMSD.svg
# python3 ../../Analysis/02_RMSD-RMSF/plot_timeseries.py analysis/RMSF.dat -o analysis/plots/02_RMSF_Profile.svg

# cd runs/18198196
# cpptraj -i ../../Analysis/02_RMSD-RMSF/RMSD_RMSF.i 

parm complex.parm7
trajin cMD-Prod.nc 1 last 10

autoimage

rms first mass out analysis/Calpha_RMSD.dat @CA
rms first mass out analysis/Backbone_RMSD.dat @CA,C,N

atomicfluct out analysis/RMSF.dat @C,CA,N byres
atomicfluct out analysis/Bfactor.dat @C,CA,N byres bfactor
