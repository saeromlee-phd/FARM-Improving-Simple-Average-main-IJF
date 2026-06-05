function [yf_c, selridge]=forecasts_eRidge_withc(y,X, org_y, org_data, seed)
    rng(seed,'twister');

    T=size(y,1);
    ydm=y;
    Xstd=X;
    N=size(X,2);
    
    f_bar=mean(X,2);
    yd=y-f_bar;

    [betap,lambdav] = pathl2(Xstd(1:T,:), yd(1:T,1));
    id = select_lambda_ridge(Xstd(1:T,:), yd(1:T,1), lambdav, seed);
    beta=betap(id,:)';
    ind=(beta ~= 0) & ~isnan(beta);

    avg_beta=1/N;
    
    beta = beta + avg_beta;

    alpha=mean(ydm)-mean(Xstd)*beta;
    beta_withc=[alpha; beta];
    org_data_withc=[1,org_data(end,:)]
    yf_c=org_data_withc*beta_withc;
  
    selridge=NaN(size(beta,1),1);
    selridge(ind)=1;
 
return