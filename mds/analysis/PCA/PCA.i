parm complex.parm7
trajin NAME_combine.nc
rms first :1-194&!@H=
average crdset NAME-average
createcrd NAME-trajectories
run
crdaction NAME-trajectories rms ref NAME-average :1-194&!@H=
crdaction NAME-trajectories matrix covar name NAME-covar :1-194&!@H=
runanalysis diagmatrix NAME-covar out NAME-evecs.dat vecs 10 name myEvecs nmwiz nmwizvecs 10 nmwizfile NAME.nmd nmwizmask :1-194&!@H=
crdaction NAME-trajectories projection Mode modes myEvecs out projection.txt beg 1 end 3 :1-194&!@H= crdframes 1,last
hist Mode:1 bins 100 out NAME-hist.agr norm name Mode-1
hist Mode:2 bins 100 out NAME-hist.agr norm name Mode-2
hist Mode:3 bins 100 out NAME-hist.agr norm name Mode-3
hist Mode:1 Mode:2 bins 100 out hists_1-2.gnu name PC12 free 300
hist Mode:1 Mode:2 bins 100 out hists_1-3.gnu name PC13 free 300
hist Mode:1 Mode:2 bins 100 out hists_2-3.gnu name PC23 free 300
run
clear all
readdata NAME-evecs.dat name Evecs
parm complex.parm7
parmstrip !(:1-194&!@H=)
parmwrite out NAME-modes.parm7
runanalysis modes name Evecs trajout NAME-mode1.nc pcmin -100 pcmax 100 tmode 1 trajoutmask :1-194&!@H= trajoutfmt netcdf
