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
trajin SMT-AMB_combine.nc 1 last 1
distance d1 :354 :68,69,70,84,91,87 out d1.dat
angle a1 :68,69,70, :354 :84,91,87 out a1.dat
go
calcstate state Groove,d1,8,17,a1,0,25 \
state Partial,d1,5,16,a1,26,95 \
state Intercalation,d1,0,4,a1,95,200 \
name DDD out \ 
calcstate.svt.dat \
curveout calcstate.curve.agr \
stateout calcstate.states.dat \
transout calcstate.trans.dat \
countout calcstate.count.dat 
go
