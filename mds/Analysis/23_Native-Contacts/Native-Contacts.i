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

parm complex-No-WAT.parm7
trajin Name-No-WAT_combine.nc

nativecontacts name NC1 :1-675&!@H= :676-981&!@H= \   
   writecontacts native-contacts.dat \   
   resout resout.dat \   
   distance 3.0 \
   byresidue out all-residues.dat mindist maxdist \
   map mapout gnu \
   contactpdb contactspdb.pdb \
   series seriesout native-contacts-series.dat
run
lifetime NC1[NC] out lifetime.dat
run 
