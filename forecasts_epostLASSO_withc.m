function [yfelasso_c, selelasso]=forecasts_epostLASSO_withc(y,X, org_y, org_data, seed)
 
    rng(seed,'twister');

    f_bar=mean(X,2);
    yd=y-f_bar;
    
    T=size(X,1);
    N=size(X,2);
    [delta,lambdav] = pathl1(X(1:T,:), yd(1:T,1));
    id = select_lambda_lasso(X(1:T,:), yd(1:T,1), lambdav, seed);
    beta=delta(id,:)';
    indd=(beta ~= 0) & ~isnan(beta);
    deltapost=X(:,indd)\yd;

    nonzero_idx = find(beta ~= 0);
    beta(nonzero_idx(1:length(deltapost))) = deltapost;
    avg_beta=1/N;
    
    beta = beta + avg_beta;
    alpha=mean(y)-mean(X)*beta
    yfelasso_c=alpha+beta'*org_data(:,:)';
    
    ind=(beta ~= 0) & ~isnan(beta);
    selelasso=NaN(size(beta,1),1);
    selelasso(ind)=1;
    return