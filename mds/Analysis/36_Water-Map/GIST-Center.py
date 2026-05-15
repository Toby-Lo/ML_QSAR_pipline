import numpy as np

class Atom:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

def read_atoms(pdb_file, residue_id):
    atoms = []
    residue_id = residue_id.strip()
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                res_id = line[22:26].strip()
                if res_id == residue_id:
                    try:
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                        atoms.append(Atom(x, y, z))
                    except ValueError:
                        continue
    return atoms

def compute_centroid_and_box(atoms, spacing=0.5, padding=5.0):
    coords = np.array([[a.x, a.y, a.z] for a in atoms])
    centroid = coords.mean(axis=0)

    min_bounds = coords.min(axis=0) - padding
    max_bounds = coords.max(axis=0) + padding

    dims = np.ceil((max_bounds - min_bounds) / spacing).astype(int)
    return centroid, dims

def write_gist_input(centroid, dims, res_id, frame_number, traj_file, output_file="GIST.in"):
    with open(output_file, 'w') as f:
        f.write("parm complex.parm7\n")
        f.write(f"trajin {traj_file} 1 {frame_number} 1\n")
        f.write(f"rms first :1-{int(res_id)-1}@CA,C,N,O\n")
        f.write("strip :NA,:Cl\n\n")
        f.write("gist doorder refdens 0.0335 temp 300.0 \\\n")
        f.write(f"     gridcntr {centroid[0]:.3f} {centroid[1]:.3f} {centroid[2]:.3f} \\\n")
        f.write(f"     griddim {dims[0]} {dims[1]} {dims[2]} \\\n")
        f.write("     gridspacn 0.5 \\\n")
        f.write(f"     out gist_residue{res_id}.dat\n")
        f.write("go\n")

# --- Interactive Inputs ---
pdb_file = input("Enter the PDB filename (e.g., frame5500.pdb): ").strip()
traj_file = input("Enter the trajectory filename (e.g., systemX_1.nc): ").strip()
res_id = input("Enter the ligand residue number (e.g., 441): ").strip()
frame_number = input("Enter the frame number used to generate this PDB (e.g., 5500): ").strip()

atoms = read_atoms(pdb_file, res_id)
if not atoms:
    print(f"❌ No atoms found for residue {res_id}. Please check the file and try again.")
else:
    centroid, dims = compute_centroid_and_box(atoms)
    write_gist_input(centroid, dims, res_id, frame_number, traj_file)
    print("\n✅ GIST.in generated successfully.")
    print(f"Trajectory: {traj_file}")
    print(f"Grid Center: {centroid}")
    print(f"Grid Dimensions: {dims}")
