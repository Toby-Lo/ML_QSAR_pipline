Sample input file for GB and PB calculation
&general
startframe=9001, endframe=10000, interval=2, ligand_mask=:729, receptor_mask=:1-728, keep_files=1, netcdf=1, verbose=2,
/
&gb
igb=8, saltcon=0.150,
/
&pb
inp=1, istrng=0.15, radiopt=0,
/
