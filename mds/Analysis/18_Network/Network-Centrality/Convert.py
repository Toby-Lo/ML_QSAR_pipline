import mdtraj as md

# Load full trajectory
traj = md.load("7JXP_combine.nc", top="complex.parm7")

# Remove water and ions (WAT, NA, CL), keep protein + ligand
# This assumes your ligand is anything not water/ions and not protein
exclude_residues = ["WAT", "NA", "CL"]
selection = traj.topology.select(
    "protein or (not water and not resname NA and not resname CL)"
)

# Slice the trajectory
clean_traj = traj.atom_slice(selection)

# Save cleaned outputs
clean_traj.save_dcd("complex.dcd")
clean_traj[0].save_pdb("complex.pdb")
clean_traj[0].save_xyz("initial.xyz")
clean_traj[-1].save_xyz("final.xyz")

print("Cleaned trajectory and files saved.")
