clear all;

seed = 123;
rng(seed, 'twister');

%% Load the data based on the target variables (RPI, CPIAUCSL, PECPI, INDRPO, and UNRATE).
% Here, we use CPIAUCSL as the target variable.
data_all=readtable('2019-12-NoCPIAUCSL.xlsx'); % predictors x
real=readtable('2019-12-CPIAUCSL.xlsx'); % forcast target y

data_all=table2array(data_all);
real=table2array(real);

% data without transform code
data=[data_all(2:end,:)];

% data transformations
% transform data based on the transformation codes
data=mtrans(data,[data_all(1,:)]);
data(:, any(isnan(data), 1)) = [];

real=mtrans(real(2:end),real(1,1));

T=size(data,1); % time dimension
N=size(data,2); % the number of predictors
h=1; % forecast horizon (h=1,2,3)

%% Set the sample period
R=118; % training period T1
Q=120; % out-of-sample period T3
P=T-R-Q; % validation period T2

%% Use the training period to construct forecasts
for t=1:T-R;
    sample_est=t:t+R-1-h;
    sample_est_y=t+h:t+R-1;
    for i=1:N
        data_sample=data(sample_est,:);
        c=ones(R-h,1);
        data_sample_c=[c,data_sample(:,i)];
        real_sample=real(sample_est_y,:);
        bi_real=data_sample_c\real_sample;

        % forecasts
        data_c=[1,data(t+R-h,i)];
        fi_real(t,i)=data_c*bi_real;
    end
end

%% Calculate the average of all potential forecasts
f0_real=mean(fi_real,2);

%% Construct combined forecasts
wd=P; % Define rolling window. 
ep=T-wd-R; 

% 0. Simple average forecast 
for t=1:ep
   % get subsample
   s_start=t;
   s_end=t+wd-1;
  
   fcsts_simple(t,:)=f0_real(s_end+1,:);
end

% 1. FARM2 (The Factor-Adjusted Regularized Model introduced in the Online Supplementary Appendix, including additional factors via PCA)
nm=4; 
for t=1:ep;
    % get subsample
    s_start=t;
    s_end=t+wd-1;
    f0_all_real_sample = f0_real(s_start:s_end,:);
    real_sample = real(s_start+R:s_end+R,:);
    di=fi_real(s_start:s_end,:)-f0_all_real_sample;
    di_oos=fi_real(s_end+1,:)-f0_real(s_end+1,:);

    % Extract factors in di
    pmax=2; % maximum # of factors

    % Extract factors 
    [G,lambda]=panelFactorNew(di,pmax);
    etahat=G\real_sample;
    sel_phat(t,:)=size(G,2)
    sel_etahat= etahat';
    sel_factor= G;
    sel_lambda = lambda;
    factor_di_oos = [];
    vi_PCA_sample = di - sel_factor * sel_etahat';

    for l=1:size(sel_lambda,2);
        ar_model = fitlm(1:wd, sel_factor(:,l));         
        factor_di_oos = cat(2,factor_di_oos, predict(ar_model, wd+t)* sel_etahat(:, l)');
        vi_PCA_sample_oos = di_oos - predict(ar_model, wd+t)*sel_etahat'
    end

    factor_di = sel_factor* sel_etahat'
  
    f0_PCA = sum(factor_di,2)
    f0_PCA_oos = sum(factor_di_oos,2)
    all_factor = f0_real(s_start:s_end,:)+f0_PCA;
    u0_PCA_real_sample = real_sample - all_factor;
    all_factor_oos = f0_real(s_end+1,:)+f0_PCA_oos;
    fcst(t,:)=all_factor_oos;

    [fcsts_FARM2_real(t,:)]=forecasts_FARM_withc(u0_PCA_real_sample, di, all_factor_oos, di_oos, all_factor_oos, seed);
end

% 2. FARM 3 (The Factor-Adjusted Regularized Model introduced in the Online Supplementary Appendix, including additional factors via SPCA)
for t=1:ep;
    % get subsample

    s_start=t;
    s_end=t+wd-1;
    f0_all_real_sample = f0_real(s_start:s_end,:);
    real_sample = real(s_start+R:s_end+R,:);
    di=fi_real(s_start:s_end,:)-f0_all_real_sample;
    di_oos=fi_real(s_end+1,:)-f0_real(s_end+1,:);

    % Extract factors in di
    pmax=2; % maximum # of factors

    % Tuning parameters
    tuningrange_SAVG = floor(0.1*N):1:N; % tuning range for SPCA
    param_savg.pmax = pmax; param_savg.dt = di; param_savg.yt = real_sample;
    
    SPCAres = kfoldcv_SPCA(3,3,@supervisedPCA,param_savg,tuningrange_SAVG, seed);
    sel_etahat= SPCAres.etahat;
    sel_factor= SPCAres.Fhat;
    sel_lambda = SPCAres.Index;
    factor_di=zeros(wd,1);
    factor_di_oos=zeros(1,1);
    vi_SPCA_sample = [];
    vi_SPCA_sample_oos = [];
    factor_di_oos = [];
    fi_real_oos = [];

    for l=1:size(sel_lambda,2);
        col_idx = find(sel_lambda(:, l) ~= 0);
        vi_each = (di(:, col_idx) - factor_di) - sel_factor(:,l) * sel_etahat(:, l)';
        ar_model = fitlm(1:wd, sel_factor(:,l));         
        factor_di_oos = cat(2,factor_di_oos, predict(ar_model, wd+t)* sel_etahat(:, l)');
        vi_each_oos = (di_oos(:, col_idx) - factor_di_oos) - predict(ar_model, wd+t)*sel_etahat(:, l)'
        for m = 1:length(col_idx);
            di(:,col_idx(m)) = di(:,col_idx(m))-sel_factor(:,l) * sel_etahat(:, l)'
            di_oos(:,col_idx(m)) = di_oos(:,col_idx(m))-predict(ar_model, wd+t)*sel_etahat(:, l)'
        end
        factor_di_each = sel_factor(:,l) * sel_etahat(:, l)'
        vi_SPCA_sample = cat(2, vi_SPCA_sample, vi_each);  
        vi_SPCA_sample_oos = cat(2,vi_SPCA_sample_oos,vi_each_oos)
        factor_di = cat(2, factor_di, factor_di_each);
        fi_real_oos = cat(2, fi_real_oos, fi_real(s_end+1,col_idx)) 
    end

    f0_SPCA = sum(factor_di,2)
    f0_SPCA_oos = sum(factor_di_oos,2)
    all_factor = f0_real(s_start:s_end,:)+f0_SPCA;
    u0_SPCA_real_sample = real_sample - all_factor;
    all_factor_oos = f0_real(s_end+1,:)+f0_SPCA_oos;
    fcst(t,:)=all_factor_oos;

    [fcsts_FARM3_real(t,:)]=forecasts_FARM_withc(u0_SPCA_real_sample, di, all_factor_oos, di_oos, all_factor_oos, seed);
end

%% Combined Forecast Matrix 
% Rows: Time periods (t), Columns: Different combination methods (Simple Average, FARM2, and FARM3)
fcsts_all_oos=[fcsts_simple, fcsts_FARM2_real, fcsts_FARM3_real];

%% evaluate forecasting performance
% compute relative RMSFE (vs Simple average combination benchmark)
benchmark=1;  % the first method (the simple averaege) is the benchmark
 
[rmsfe_real, relative_rmsfe_real, ereal]=compute_rmsfe(real(R+wd+1:end,1), fcsts_all_oos, benchmark);
[msfe_real, relative_msfe_real, ereal]=compute_msfe(real(R+wd+1:end,1), fcsts_all_oos, benchmark);


%% Diebold and Mariano Test 
for m = 1:size(fcsts_all_oos, 2) - 1
    % forecast errors
    e_simple = real(R+wd+1:end,1) - fcsts_all_oos(:, 1); % forecast errors from the simple average
    e_comb = real(R+wd+1:end,1) - fcsts_all_oos(:, m + 1); % forecast errors from other combined forecasts
    
    % squared forecast errors
    e_0_sq = e_simple .* e_simple; 
    e_1_sq = e_comb .* e_comb;
    
    % e_0_sq - e_1_sq 
    diff = e_0_sq - e_1_sq;
    
    % t-test: H0: E(diff) = 0, H1: E(diff) > 0 
    [h, p_value(m, 1)] = ttest(diff, 0, 'Tail', 'right');
end

% save results
results = {'fcsts_real', 'relative_msfe_real', ' p_value'; fcsts_all_oos, relative_msfe_real,  p_value};
fileName = 'Appendix_Results_CPIAUCSL1.xlsx'; % Change the file name based on the forecast target and horizon ($h$). For example, Appendix_Results_CPIAUCSL1 denotes the results for the target variable CPIAUCSL with $h=1$ in Table A1.
% save Excel
for i = 1:3
    sheetTitle = results{1, i};
    sheetData = results{2, i};
    writematrix(sheetData, fileName, 'Sheet', sheetTitle);  
    
end

