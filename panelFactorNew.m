% X is an input matrix (T by N)   (not necessaarily the regressor matrix)
% r is the number of true factors
% output: factor (F) and factor loadings lambda (L)
% restriction: either F'F/T=I  or L'*L/N=I, depending if T<N,  but F*L is
% the same 

% VNT: diagonal matrix of eigenvalues

function [factor, lambda, VNT]=panelFactorNew(X, r);
[T,N]=size(X);
 if T < N ;
   XX=X*X'/(N*T);
  [Fhat0,eigval,Fhat1]=svd(XX);
  factor=Fhat0(:,1:r)*sqrt(T);
  lambda=X'*factor/T;
  VNT=eigval(1:r,1:r);
 
else

  XX=X'*X/(N*T);
  [Fhat0,eigval,Fhat1]=svd(XX);
  lambda=Fhat0(:,1:r)*sqrt(N);
  factor=X*lambda/N;
  VNT=eigval(1:r,1:r); 
end
