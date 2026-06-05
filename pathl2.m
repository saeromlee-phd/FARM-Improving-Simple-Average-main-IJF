% pathl1.m

function [betamat lambdav Xsca0 XXmat0 cvec0] = pathl2(X, y, varset, K, maxiter, tol)

[n p] = size(X);
maxsize = min(ceil(n/2), p); %maxsize = min(ceil(n/log(n)), p);

if (nargin < 3) varset = []; end
if (nargin < 4) K = 50; end
if (nargin < 5) maxiter = 50; end
if (nargin < 6) tol = 1e-4; end

%newvarord = randperm(p); % randomly permute variables to ensure that no 
                         % information on variable ordering is used in
                         % coordinate optimization (e.g., when true 
                         % variables are first in order)
                         
newvarord=[1:p];
                         
X = X(:, newvarord);
[dum ind] = sort(newvarord, 'ascend');
varset = ind(varset);

% rescale X columnwise 
Xsca = sqrt(sum(X.^2))/sqrt(n);
X = X./(ones(n, 1)*Xsca);

XXmat = n^(-1)*X'*X;
cvec = n^(-1)*X'*y;

% set grid of lambda
lamvec = logspace(-5, 6, 100); 

% initialization
betap = zeros(length(lamvec), p);

for i = 1:length(lamvec)
    if (i == 1)
        inival = betap(i, :)';
    else
        inival = betap(i - 1, :)';
    end
    lambda = lamvec(i);
    beta = ol2(XXmat, cvec, inival, lambda, varset, maxiter, tol);
    betap(i, :) = beta;
    varset = union(find(beta), varset);
    
%     if (sum(beta ~= 0) > maxsize)
%         lamvec(i:end) = [];
%         betap(i:end, :) = [];
%         break
%     end
end

% rescale beta coefficients to original scales
for i = 1:length(lamvec)
    betap(i, :) = betap(i, :)./Xsca;
end
[dum ind] = sort(newvarord, 'ascend');
betap = betap(:, ind); % back to original order of variables

if (nargout == 1)
    betamat = betap;
elseif (nargout == 2)
    betamat = betap;
    lambdav = lamvec;
elseif (nargout == 3)
    betamat = betap;
    lambdav = lamvec;
    Xsca0 = Xsca;
elseif (nargout == 4)
    betamat = betap;
    lambdav = lamvec;
    Xsca0 = Xsca;
    XXmat0 = XXmat;
elseif (nargout == 5)
    betamat = betap;
    lambdav = lamvec;
    Xsca0 = Xsca;
    XXmat0 = XXmat;
    cvec0 = cvec;
end
