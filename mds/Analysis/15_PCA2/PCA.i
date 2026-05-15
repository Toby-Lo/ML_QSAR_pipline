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
trajin 6Z12-LIG_combine.nc

autoimage anchor @CA
rms first @N,C,CA
average @N,C,CA crdset trajaverage
createcrd trajectory
run
crdaction trajectory rms ref trajaverage @N,C,CA
crdaction trajectory matrix covar name covmatrix @N,C,CA
runanalysis diagmatrix covmatrix out evecs.dat vecs 10 name eigenvectors nmwiz nmwizvecs 10 nmwizfile normalmodes.nmd nmwizmask @N,C,CA
crdaction trajectory projection Mode modes eigenvectors beg 1 end 10 @N,C,CA out pca.dat crdframes 1,last
hist Mode:1 bins 100 out Name-hist.agr norm name Mode-1
hist Mode:2 bins 100 out Name-hist.agr norm name Mode-2
hist Mode:3 bins 100 out Name-hist.agr norm name Mode-3
hist Mode:1 Mode:2 bins 100 out hists_1-2.gnu name PC12 free 300
hist Mode:1 Mode:2 bins 100 out hists_1-3.gnu name PC13 free 300
hist Mode:1 Mode:2 bins 100 out hists_2-3.gnu name PC23 free 300
run
clear all
readdata evecs.dat name eigenvectors
parm complex.parm7
parmwrite out modes.parm7
runanalysis modes name eigenvectors trajout modes01.nc pcmin -150 pcmax 150 \
tmode 1 trajoutmask @N,C,CA trajoutfmt netcdf
run


#python colormap.py pca.dat PCA pca.png
