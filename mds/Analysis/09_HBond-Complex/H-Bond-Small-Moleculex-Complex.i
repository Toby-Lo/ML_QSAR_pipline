# === Usage (run from a simulation directory) ===
#   cd runs/<SYS>
#   mkdir -p analysis
#   cpptraj -i ../../Analysis/09_HBond-Complex/H-Bond-Small-Moleculex-Complex.i
#
# === What to edit (common) ===
# - Topology: `parm complex.parm7` (only change if you renamed it)
# - Trajectory: `trajin cMD-Prod*.nc ...` (use your real trajectory; centered/imaged is recommended)
# - Masks: ligand residue name/number (e.g., `:UNL`), protein range (e.g., `:1-XXX`)
# - Sampling: `trajin ... 1 last 10` controls stride (here: every 10th frame)
# ==============================================
# Scope:
# - This script computes STRICT protein-ligand H-bonds using two explicit masks.
# - Update masks below for each system if residue numbering differs.
#   receptor mask (protein): :1-223
#   ligand mask           : :224

# Plotting:
# python3 ../../Analysis/09_HBond-Complex/plot_timeseries.py analysis/HBond_PL.gnu -o analysis/plots/09_HBond_PL_TimeSeries.svg


parm complex.parm7
trajin cMD-Prod.nc 1 last 10
autoimage

# Protein-ligand H-bonds only (strict, directional decomposition).
# 1) Protein donor -> Ligand acceptor
hbond PL_p2l out analysis/HBond_PL_p2l.hbvtime.dat \
  donormask :1-221 acceptormask :UNL \
  avgout analysis/HBond_PL_p2l.avg.dat series uuseries analysis/HBond_PL_p2l.gnu nointramol

# 2) Ligand donor -> Protein acceptor
hbond PL_l2p out analysis/HBond_PL_l2p.hbvtime.dat \
  donormask :UNL acceptormask :1-221 \
  avgout analysis/HBond_PL_l2p.avg.dat series uuseries analysis/HBond_PL_l2p.gnu nointramol

# First run: generate HBond data sets from trajectory.
run

# Then perform lifetime analysis on generated data sets.
lifetime PL_p2l[solutehb] out analysis/HBond_PL_p2l_lifetime.dat
lifetime PL_l2p[solutehb] out analysis/HBond_PL_l2p_lifetime.dat
runanalysis


#Using the bash command: "sort -g contacts-lifetime.dat -k5" to sort the file using the 5th column (lifetime frames) returns:
