ante-MMPBSA.py -p complex.parm7 -c complex_No_WAT.parm7 -r protein.parm7 -l ligand.parm7 -s ":WAT,NA,CL" -m ':1-589' --radii=mbondi2




MMPBSA.py.MPI -O -i MMPBSA-GBSA.i -o Name-FINAL_RESULTS_MMPBSA.dat -sp complex.parm7 -cp complex_No_WAT.parm7 -rp protein.parm7 -lp ligand.parm7 -y Name_combine.nc







export DO_PARALLEL='mpirun -np 40'
