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

# Load the topology file
parm complex.parm7

# Load the trajectory file
trajin Name_combine.nc

# Calculate the RDF
# The solvent mask is defined by atom indices 196 to 10334 (water molecules)
# The solute mask is defined by residues 1 to 95 (your solute)
# The 'spacing' and 'maximum' values for 100 bins need to be determined based on the maximum distance you are interested in.
# For example, if the maximum distance is 10 Å, and you want 100 bins, the spacing would be 0.1 Å (10 Å / 100 bins).
radial out Name-RDF.txt 0.1 10.0 :195-10334 :1-194 noimage density
run
