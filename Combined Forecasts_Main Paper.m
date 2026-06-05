clear all;

seed = 123;
rng(seed, 'twister');

%% Load the data based on the target variables (RPI, CPIAUCSL, PECPI, INDRPO, and UNRATE)
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

% 1. AR(1)
for t=1:ep
    % get subsample
    s_start=t+R;
    s_end=t+wd-1+R;

    y_train = real(s_start:s_end,:); 

    % AR(1) model
    model = ar(y_train, 1); 
    fcsts_AR(t,:) = model.a(2) * real(s_end,:); 
end

% 2. GR (The OLS regression proposed by Granger and Ramanathan (1984))
for t=1:ep;
    % get subsample
    s_start=t;
    s_end=t+wd-1;

    fi_real_sample=fi_real(s_start:s_end,:);
    real_sample=real(s_start+R:s_end+R,:);

    beta_gr_real=fi_real_sample\real_sample;
    alpha_gr=mean(real_sample)-mean(fi_real_sample)*beta_gr_real;
    fcsts_gr_real(t,:)=alpha_gr+fi_real(s_end+1,:)*beta_gr_real;   

end


% 3. DS (The egalitarian method from Diebold and Shin (2019))
for t=1:ep;
    % get subsample
    s_start=t;
    s_end=t+wd-1;

    fi_real_sample=fi_real(s_start:s_end,:);
    real_sample=real(s_start+R:s_end+R,:);
    fi_real_oos=fi_real(s_end+1,:);
  
    [fcsts_eLasso_real(t,:), sel_all_eLasso_real(:,:,t)]=forecasts_epostLASSO_withc(real_sample, fi_real_sample, real_sample, fi_real_oos, seed);
    [fcsts_eAlasso_real(t,:), sel_all_eAlasso_real(:,:,t)]=forecasts_epostALASSO_withc(real_sample, fi_real_sample, real_sample, fi_real_oos, seed);
    [fcsts_eRidge_real(t,:), sel_all_eRidge_real(:,:,t)]=forecasts_eRidge_withc(real_sample, fi_real_sample, real_sample, fi_real_oos, seed);
    [fcsts_eBoosting_real(t,:), sel_all_eBoosting_real(:,:,t)]=forecasts_eBoosting_withc(real_sample, fi_real_sample, real_sample, fi_real_oos, seed);

end

% 4. P-DS (The partially egalitarian from Diebold and Shin (2019))
for t=1:ep
    % get subsample
    s_start=t;
    s_end=t+wd-1;

    fi_real_sample=fi_real(s_start:s_end,:);
    real_sample=real(s_start+R:s_end+R,:);
    fi_real_oos=fi_real(s_end+1,:);

    [betap,lambdav] = pathl1(fi_real_sample, real_sample);
    id = select_lambda_lasso(fi_real_sample,real_sample, lambdav, seed);
    beta_pre=betap(id,:)';

    % non-zero beta
    idx = beta_pre ~= 0; 
    nm_nonbeta=sum(beta_pre~= 0);

    fi_real_sample_screen = fi_real_sample(:, idx);  
    fi_real_oos_screen=fi_real(s_end+1,idx);

    [fcsts_peLasso_real(t,:)]=forecasts_epostLASSO_withc(real_sample, fi_real_sample_screen, real_sample, fi_real_oos_screen, seed);
    [fcsts_peAlasso_real(t,:)]=forecasts_epostALASSO_withc(real_sample, fi_real_sample_screen, real_sample, fi_real_oos_screen, seed);
    [fcsts_peRidge_real(t,:)]=forecasts_eRidge_withc(real_sample, fi_real_sample_screen, real_sample, fi_real_oos_screen, seed);
    [fcsts_peBoosting_real(t,:)]=forecasts_eBoosting_withc(real_sample, fi_real_sample_screen, real_sample, fi_real_oos_screen, seed);

end

% 5. FARM1 (The Factor-Adjusted Regularized Model introduced in this paper, excluding additional factors)
nm=4; 
% initialize some variables
fcsts_FARM1_real=NaN(ep,nm);
k=size(data,2);
sel_all_FARM1_real=NaN(k,nm,ep);

for t=1:ep
    % get subsample
    s_start=t;
    s_end=t+wd-1;

    % forecasts
    fi_real_sample=fi_real(s_start:s_end,:)
    f0_real_sample=f0_real(s_start:s_end,:)
    real_sample=real(s_start+R:s_end+R,:);
   
    % u_0
    u0_real_sample=real_sample-f0_real_sample;
   
    % derive idiosyncratic components (di=f0-fi)
    f0_real_rep=repmat(f0_real_sample, 1, N);
    di_real_minus=fi_real_sample-f0_real_rep;
   
    di_real_sample=di_real_minus;
    c=ones(wd,1);
    di_real_sample_withc=[c,di_real_sample]

    f0_oos=f0_real(s_end+1,:)
    fi_oos=fi_real(s_end+1,:)
    fi_oos_withc=[1,fi_oos]

   [fcsts_FARM1_real(t,:), sel_all_FARM1_real(:,:,t)]=forecasts_FARM1_withc(u0_real_sample, di_real_sample, f0_oos, fi_oos, f0_oos, seed);

end

% 6. NL (Non-linear model via Gradient Boosting)
for t = 1:ep
    % get subsample
    s_start=t+R;
    s_end=t+wd-1+R;

    y_train = real(s_start:s_end,:); 
    x_train = fi_real(s_start-R : s_end-R, :);

    % Gradient Boosting
    model = fitrensemble(x_train, y_train, ...
        'Method','LSBoost', ...
        'NumLearningCycles',50, ...
        'LearnRate',0.001, ...
        'Learners',templateTree('MaxNumSplits',5));

    x_test = fi_real(s_end-R+1, :);
    fcsts_NL(t,:) = predict(model, x_test);
end

%% Combined Forecast Matrix 
% Rows: Time periods (t), Columns: Different combination methods (Simple Average, AR, GR, DS, P-DS, FARM1, and NL)
fcsts_all_oos=[fcsts_simple, fcsts_AR, fcsts_gr_real, fcsts_eLasso_real, fcsts_eAlasso_real,  fcsts_eBoosting_real, fcsts_eRidge_real, fcsts_peLasso_real, fcsts_peAlasso_real,  fcsts_peBoosting_real, fcsts_peRidge_real, fcsts_FARM1_real, fcsts_NL]

%% Evaluate forecasting performance
% Compute the relative RMSFE against the simple average benchmark
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
fileName = 'Results_CPIAUCSL1.xlsx'; % Change the file name based on the forecast target and horizon ($h$). For example, Results_CPIAUCSL1 denotes all results for the target variable CPIAUCSL with $h=1$ in Table 1.
% save Excel
for i = 1:3
    sheetTitle = results{1, i};
    sheetData = results{2, i};
    writematrix(sheetData, fileName, 'Sheet', sheetTitle);  
end

