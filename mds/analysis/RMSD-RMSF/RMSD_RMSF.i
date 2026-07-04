parm complex.parm7
trajin Name_combine.nc 1 last 1
autoimage
rms first @CA
rms first mass out Name_Calpha_RMSD.dat @CA
rms first mass out Name_Backbone_RMSD.dat @CA,C,N
atomicfluct out Name_RMSF.dat @C,CA,N byres
atomicfluct out Name_Bfactor.dat @C,CA,N byres bfactor

