function [yfeAlasso_c, seleAlasso]=forecasts_epostALASSO_withc(y,X, org_y, org_data, seed)
   
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
    
    % next adaptive Lasso
    Xa=X(:,ind);
    org_dataa=org_data(:,ind);
    if sum(ind)>0
       [delta_a,lambdav] = pathl1_ns(Xa(1:T,:), yd(1:T,1));
       % prediction-based selection
       id = select_lambda_lasso(Xa(1:T,:), yd(1:T,1), lambdav, seed);
       betaa=delta_a(id,:)';

       indda=(betaa ~= 0) & ~isnan(betaa);
       deltaposta=X(:,indda)\yd;
       nonzero_idx = find(betaa ~= 0);
       betaa(nonzero_idx(1:length(deltaposta))) = deltaposta;
       avg_beta=1/N;

       betaa=betaa+avg_beta;
       alphaa=mean(y)-mean(Xa)*betaa
    
       inda=abs(betaa)>0;
       n=size(X,2);    
       od=[1:n]';    
       ods1=od(ind);
       ods2=ods1(inda);
    
       indaf=NaN(n,1);
       indaf(ods2)=1;

       yfeAlasso_c=alphaa+betaa'*org_dataa(end,:)';
      
       seleAlasso=indaf;
 % report results
    else
        yfeAlasso_c=yfelasso_c;
        seleAlasso=selelasso;
    end
    return