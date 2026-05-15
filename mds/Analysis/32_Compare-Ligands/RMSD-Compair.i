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

parm reference.pdb [average]
reference reference.pdb parm [average]
parm protein-ligand1.pdb [ligand1]
trajin protein-ligand1.pdb parm [ligand1]
parm protein-ligand2.pdb [ligand2]
trajin protein-ligand2.pdb parm [ligand2]
parm protein-ligand3.pdb [ligand3]
trajin protein-ligand3.pdb parm [ligand3]
parm protein-ligand4.pdb [ligand4]
trajin protein-ligand4.pdb parm [ligand4]
parm protein-ligand5.pdb [ligand5]
trajin protein-ligand5.pdb parm [ligand5]
parm protein-ligand6.pdb [ligand6]
trajin protein-ligand6.pdb parm [ligand6]
rmsd All-atoms :1-156 reference :1-156 out rms.dat
rmsd Backbone :1-156@C,N,O,CA,CB reference :1-156@C,N,O,CA,CB out rms.dat
go 
