% ol1.m

function beta = ol1(XXmat, cvec, inival, lambda, varset, maxiter, tol)

p = length(cvec);
% XXmat = n^(-1)*X'*X
% cvec = n^(-1)*X'*y

if (nargin < 5) varset = []; end
if (nargin < 6) maxiter = 50; end
if (nargin < 7) tol = 1e-4; end

% first round of iteration
beta = inival;
iter = 1;
update = 1;
ind = 1:p;
varset = union(find(inival), varset);

while (iter <= maxiter) & (update > tol)
    iter = iter + 1;
    betaold = beta;
    
    for k = 1:length(ind)
        setr = ind;
        I = setr(k);
        setr(k) = [];
        
        % solve min_beta  (2 Lam)^(-1) (z - beta)^2 + p_lambda(|beta|)
        % for scalar beta
        z1 = XXmat(I, I);
        Lam = 1/z1;
        if isempty(setr)
            z = cvec(I)/z1;
        else
            z = (cvec(I) - XXmat(I, setr)*beta(setr))/z1;
        end
        beta(I) = ul1(z, Lam, lambda);
    end
    
    ind = find(beta);
    update = norm(beta - betaold);
    
    setr = setdiff(1:p, ind);
    if isempty(setr)
        resc = 0;
    else
        resc = abs(cvec(setr) - XXmat(setr, ind)*beta(ind));
    end
    
    indm = find(resc > lambda);
    ind = union([setr(indm)'; ind], varset);
end
