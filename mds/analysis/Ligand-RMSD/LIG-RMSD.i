parm complex.parm7 
trajin Name_combine.nc 1 1
reference Name_combine.nc [firstframe]
loadcrd Name_combine.nc prodrun
crdaction prodrun rms reference @CA,C,N
crdaction prodrun rmsd lig_rmsd ":6AD & !@/H" first out Name-lig_rmsd.dat nofit
go
