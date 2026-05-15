import numpy as np

# Load BC values
bc = np.loadtxt("complex_BC_avg.dat")

# Load the original PDB
with open("NAME_complex.pdb", "r") as f:
    lines = f.readlines()

# Update B-factor for CA atoms per residue
residue_index = -1
new_lines = []
for line in lines:
    if line.startswith("ATOM") and line[13:15] == "CA":
        residue_index += 1
        bf = bc[residue_index] if residue_index < len(bc) else 0.0
        newline = line[:60] + f"{bf:6.2f}" + line[66:]
        new_lines.append(newline)
    else:
        new_lines.append(line)

# Save new PDB
with open("complex_AV_mapped.pdb", "w") as f:
    f.writelines(new_lines)

