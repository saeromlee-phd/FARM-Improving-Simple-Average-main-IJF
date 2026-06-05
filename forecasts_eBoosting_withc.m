function [yf_c, selBoosting]=forecasts_eBoosting_withc(y,X, org_y, org_data, seed);
    rng(seed,'twister');
    
    [T,nx]=size(X);
    mupp=100;
    v=0.001; 
    I=eye(T);

    fit=zeros(T,1);
    betas=zeros(nx,mupp);
    [sel_var,fitsel,betasel]=sel_reg_b_new(y,X,nx,T);
    fit=fit+v*fitsel;
    if isempty(fit)
       fitall(:,1) = 0;
    else
       fitall(:,1) = fit;
    end
    betas(sel_var,1)=v*betasel;
    xs=X(:,sel_var);
    if isempty(xs)
       Bc=I;
    else
       Bc=I-v*(xs*xs')/sum(xs.^2);
    end
    Bm=I-Bc;
    trbm=trace(Bm);
    up=y-Bm*y;
    sig2=up'*up/T; %sigs(1,1)=sig2;
    % AIC(1,1)=log(sig2)+(1+trbm/T)/(1-(trbm+2)/T);
    AIC(1,1)=log(sig2)+trbm*log(T)/T; %BIC
     
    %iterations
    
    for jj=2:mupp
        [sel_var,fitsel, betasel]=sel_reg_b_new(y-fit,X,nx,T); %iall(j,1)=i;
        fit=fit+v*fitsel;
        if isempty(fit)
           fitall(:,jj) = 0;
        else
           fitall(:,jj) = fit;
        end 
        betas(:,jj)=betas(:,jj-1);
        betas(sel_var,jj)=betas(sel_var,jj)+v*betasel;
        xs=X(:,sel_var);
        if isempty(xs)
           Bc=I*Bc;
        else
           Bc=(I-v*(xs*xs')/sum(xs.^2))*Bc;
        end
        Bm=I-Bc;
        trbm=trace(Bm);
        up=y-Bm*y;
        sig2=up'*up/T;
        %AIC(jj,1)=log(sig2)+(1+trbm/T)/(1-(trbm+2)/T);
        AIC(jj,1)=log(sig2)+trbm*log(T)/T; %BIC
    end
    
    
    % determine which iteration is the last
    [~, it] = min(AIC);
    % select which fit it is
    fitsel = fitall(:,it);
    % compute corresponding betas
    ind = abs(betas(:,it))>0;
    selreg=X(:,ind);
    beta_x=betas(:,it);
    
    Rind=ind;
    inds=false(size(Rind,1),1);
    inds=beta_x~=0;
 
    sumbeta_x=sum(beta_x);
    alpha=mean(y)-mean(X)*beta_x;
    beta_withc=[alpha; beta_x];
    org_data_withc=[1,org_data(end,:)]
    yf_c=org_data_withc*beta_withc;

    selBoosting=NaN(nx,1);
    % assign selected regressors
    a=NaN(size(inds(1:nx)));
    a(inds(1:nx))=1;
    selBoosting(1:nx,1)=a;

    
return