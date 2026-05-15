import numpy as np
import MDAnalysis as mda

# Parameters
bc_file = "complex_BC_avg.dat"
pdb_file = "7JXP_complex.pdb"
top_n = 10  # Top N residues
output_pml = "label_top_BC.pml"

# Load BC data
bc = np.loadtxt(bc_file)
top_indices = np.argsort(bc)[-top_n:][::-1]  # Indices of top BC values

# Load structure
u = mda.Universe(pdb_file)
residues = list(u.select_atoms("name CA").residues)

resi_list = []  # Collect resi values for selection

# Write PyMOL label script
with open(output_pml, "w") as f:
    for idx in top_indices:
        res = residues[idx]
        resn = res.resname
        resi = res.resid
        resi_list.append(str(resi))
        f.write(f'label (name CA and resi {resi}), "{resn}{resi}"\n')

    # Add selection and visualization commands
    joined_resi = '+'.join(resi_list)
    f.write("\n")
    f.write(f"select Network-Cent, (name CA and resi {joined_resi})\n")
    f.write("show spheres, Network-Cent\n")
    f.write("set sphere_scale, 0.5, Network-Cent\n")
    f.write("color orange, Network-Cent\n")

print(f"✅ PyMOL label + selection script written to: {output_pml}")
