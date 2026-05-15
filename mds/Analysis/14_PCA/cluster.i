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
trajin Name_combine.nc
readdata projection.txt name pca_data

# K-means clustering on the PCA data
cluster data pca_data kmeans clusters 2 \
        out kmeans.clusters \
        summary kmeans.summary \
        rms :1-194@CA mass
run

#python3 PCA.py projection.txt PCA pca.png
