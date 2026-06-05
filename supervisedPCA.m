function res = supervisedPCA(param, seed)
% This function performs supervised simple average forecast combination

%% INPUT

% dt          is T by n matrix (potential idiosyncratic matrix)
% yt          is T by 1 target variable
% usep        is # of factors we use (1 by J vector)
% tuning      is # of test assets we use at each step

%% INITIALIZATION
rng(seed, 'twister');
if ~isfield(param,'q')
   param.q = 1;
end

dt = param.dt;
yt = param.yt;
usep = param.usep;
N0 = param.tuning;
q = param.q;

T  =  size(dt,1);
n  =  size(dt,2);
d  =  size(yt,2);
J  =  length(usep);
pmax = max(usep);    
%% ESTIMATION
Index               =      [];% is the subset we choose at each step
k                   =      0;% is # of steps
B                   =      zeros(pmax,n);

dt0                 =      dt;
yt0                 =      yt;
while(k<pmax)

    COR   = abs(corr(dt0,yt0));

    L = max(COR,[],2);
    [bb,i] = sort(L);
    if N0<n
        II = (L >= bb(n-N0));
    else
        II = L>-1;
    end

    k = k + 1;

    Index(:,k) = II;

% perform PCA
    [U,S,V] = svds(dt0(:,II),1); 
    F=U(:,:)*S(1,1); % factor
    idx = find(Index(:,k) == 1);
    Index(idx,1) = V;

    %lamdba_all(:,k) = V; % factor loadings
    etahat_all(:,k) = yt0\F;


% projection (compute the residuals)
    yt0 = yt0 - F*etahat_all(:,k)';
    dt0(:,II) = dt0(:,II) - F*V'; 
    F_all(:,k) = F;
     
end

for jj = 1:length(usep)
    phat = usep(jj);
    Fhat = F_all(:,1:phat);
    etahat = etahat_all(:,1:phat);
    fcst_error(:,jj) =mean(sum(yt0).^2);   
end
%% Results
res.fcst_error = fcst_error;
res.etahat = etahat_all;
res.Fhat = F_all;
res.Index = Index;





