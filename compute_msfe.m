function [mses,relative_msfe, e]=compute_msfe(y,fcsts, benchmark)

[Tf,nm]=size(fcsts);
%compute MSEs
e=y*ones(1,nm)-fcsts;
mses=diag(e'*e/Tf);
relative_msfe=mses/mses(benchmark);

return