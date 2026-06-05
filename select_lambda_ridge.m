function idn = select_lambda_ridge(X, y, lambdav, seed)

rng(seed, 'twister');
% prediction-based selection

[nlam] = size(lambdav,2);
T=size(y,1);
ts=T/10;
lambda_sel10=zeros(10,1);
for i=1:10 %10-fold cv
    inds=false(T,1);
    inds(floor((i-1)*ts+1):floor(i*ts),1)=true;
    valx=X(inds,:);
    valy=y(inds,1);
    trainx=X(not(inds),:);
    trainy=y(not(inds),1);
    % estimate over training sample
    [betamat] = pathl2_l(trainx, trainy,lambdav);
    % evaluate
    mse= mean((valy*ones(1, nlam) - valx*betamat').^2);
    [mseval id] = min(mse);
    lambda_sel10(i,1)=lambdav(id);
end

lambda_ave=mean(lambda_sel10);
[temp idn] = min(abs(lambdav-lambda_ave));
