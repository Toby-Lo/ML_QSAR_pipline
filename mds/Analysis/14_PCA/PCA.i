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

# PCA on protein Cα atoms (edit mask if needed).
rms first @CA
matrix covar name COVAR @CA
run

# Diagonalize covariance after matrix is accumulated.
runanalysis diagmatrix COVAR out analysis/PCA_evecs.dat vecs 10 name EVECS

# Re-read trajectory and project onto PC modes.
trajin cMD-Prod.nc 1 last 10
autoimage
rms first @CA
projection PC modes EVECS out analysis/PCA_projection.dat beg 1 end 3 @CA

# 1D histograms of PC1/2/3 projections.
hist PC:1 bins 100 out analysis/PCA_hist.agr norm name PC1
hist PC:2 bins 100 out analysis/PCA_hist.agr norm name PC2
hist PC:3 bins 100 out analysis/PCA_hist.agr norm name PC3
run
