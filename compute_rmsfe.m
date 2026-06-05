function [rmses,relative_rmsfe, e]=compute_rmsfe(y,fcsts, benchmark)

[Tf,nm]=size(fcsts);
%compute MSEs
e=y*ones(1,nm)-fcsts;
mses=diag(e'*e/Tf);
rmses=mses.^0.5;
relative_rmsfe=rmses/rmses(benchmark);


return