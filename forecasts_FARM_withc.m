function [fcsts, sel]=forecasts_FARM_withc(y, data, org_y, org_data, mean, seed) % forecasts without considering factors

rng(seed,'twister');
nm=4 % number of forecasting methods (Lasso, Adaptive Lasso, Ridge, Boosting)

if nargin==1
   fcsts=nm; % return the number of forecasting methods
   sel=0;
else

%% individual forecasting methods
   sel=NaN(size(data,2),nm);
   method_no=1;

%% (a-b) Lasso and Adaptive Lasso 
   nd=size(data,2);

  % Lasso and Adaptive Lasso forecasts
    method_noa=method_no+1;
    [fcsts(1,method_no), sellasso, fcsts(1,method_noa), sellassoa]=Lassof_fcst_post(y,data, org_y, org_data, seed); 

  % assign selected variables     
    sel(:,method_no)=sellasso;
    sel(:,method_noa)=sellassoa;
  
%% (c) L2-Boosting
   method_no=method_noa+1;
   [fcsts(1,method_no), sel(:,method_no)]=Boosting_fcst(y,data, org_y, org_data, seed);
  

%% (d) Ridge
    method_no=method_no+1;
    % Ridge forecasts
    [fcsts(1,method_no), selridge]=Ridge_fcst(y,data, org_y, org_data, seed); 
    % assign selected variables     
    sel(:,method_no)=selridge;

end

return


%% local functions

function [yf_c, sellasso, yfa_c, sellassoa]=Lassof_fcst_post(y,X, org_y, org_data, seed)
 rng(seed,'twister');   
    T=size(y,1);
    ydm=y;
    Xstd=X;

    [betap,lambdav] = pathl1(Xstd(1:T,:), ydm(1:T,1));
    id = select_lambda_lasso(Xstd(1:T,:), ydm(1:T,1), lambdav, seed);
    beta=betap(id,:)';
    ind=(beta ~= 0) & ~isnan(beta);
    betapost=Xstd(:,ind)\y;
    alpha=mean(ydm)-mean(Xstd(:,ind))*betapost;
    beta_withc=[alpha; betapost];
    if sum(ind)>0
    org_data_withc=[1,org_data(end,ind)]
    yf_x=beta_withc'*org_data_withc';
    yf_x(isnan(yf_x)) = 0;
    yf_c=org_y(end,:)+yf_x;
    else
    yf_c=org_y(end,:);
    end
    
    sellasso=NaN(size(beta,1),1);
    sellasso(ind)=1;

    % next adaptive Lasso
    Xa=Xstd(:,ind)%*diag(abs(beta(ind)));
    org_dataa=org_data(:,ind)%*diag(abs(beta(ind)));
    if sum(ind)>0
       [beta_a,lambdav] = pathl1_ns(Xa(1:T,:), ydm(1:T,1));
       % prediction-based selection
       id = select_lambda_lasso(Xa(1:T,:), ydm(1:T,1), lambdav, seed);
       betaa=beta_a(id,:)';

       inda=abs(betaa)>0;
       n=size(Xstd,2);    
       od=[1:n]';    
       ods1=od(ind);
       ods2=ods1(inda);
    
       indaf=NaN(n,1);
       indaf(ods2)=1;

       betaposta=Xa(:,inda)\y;
       alphaa=mean(ydm)-mean(Xa(:,inda))*betaposta;
       beta_withca=[alphaa; betaposta];

       org_dataa_withc=[1,org_dataa(end,inda)];
       yfa_x=beta_withca'* org_dataa_withc';
       sumbetaposta=sum(betaposta);
       yfa_x(isnan(yfa_x)) = 0;

       yfa_c=org_y(end,:)+yfa_x;
       sellassoa=indaf;
 % report results
    else
        yfa_c=yf_c;
        sellassoa=sellasso;
    end
    
return


function [yf_c, selBoosting]=Boosting_fcst(y,X, org_y, org_data, seed);
rng(seed,'twister');
    [T,nx]=size(X);
    mupp=3000;
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
    
    
    % determine which iteration is last
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
    yf_0=org_data_withc*beta_withc;


    yf_c=org_y(end,:)+yf_0;
    
    
    selBoosting=NaN(nx,1);
    % assign selected regressors
    a=NaN(size(inds(1:nx)));
    a(inds(1:nx))=1;
    selBoosting(1:nx,1)=a;

    
return

function [yf_c, selridge]=Ridge_fcst(y,X, org_y, org_data, seed)
   rng(seed,'twister'); 
    T=size(y,1);
    ydm=y;
    Xstd=X;

    [betap,lambdav] = pathl2(Xstd(1:T,:), ydm(1:T,1));
    id = select_lambda_ridge(Xstd(1:T,:), ydm(1:T,1), lambdav, seed);
    beta=betap(id,:)';
    ind=(beta ~= 0) & ~isnan(beta);
    if sum(ind)>0

       alpha=mean(ydm)-mean(Xstd)*beta;
       beta_withc=[alpha; beta];
       org_data_withc=[1,org_data(end,:)]
       yf_x=org_data_withc*beta_withc;
  
       sumbeta=sum(beta);
       yf_x(isnan(yf_x)) = 0;
       yf_c=org_y(end,:)+yf_x;
       selridge=NaN(size(beta,1),1);
       selridge(ind)=1;
    else
       yf_c=org_y(end,:);
       selridge=NaN(size(beta,1),1);
       selridge(ind)=1;
    end
return

