parm complex.parm7
trajin Name_combine.nc 1 last 1
hbond contacts :1-490 avgout Name_avg.dat series uuseries Name_hbond.gnu nointramol
lifetime contacts[all] out Name_contacts-lifetime.dat
go


#Using the bash command: "sort -g contacts-lifetime.dat -k5" to sort the file using the 5th column (lifetime frames) returns:
