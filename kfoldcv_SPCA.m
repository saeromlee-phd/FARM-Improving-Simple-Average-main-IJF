function res=kfoldcv_SPCA(M,K,func1,param,tuningrange,seed)
rng(seed, 'twister');
%% Input
% M             repeat K-fold CV M times
% K             K-fold
% func1         simple average f_bar fuction
% param         parameters for func1
% tuningrange

if ~isfield(param,'pmax')
   param.pmax = 1; 
end

yt = param.yt;
dt = param.dt;
pmax = param.pmax;

T = size(dt,1);
tsr2 = zeros(K,M,length(tuningrange),param.pmax);

tr = length(tuningrange);

for m = 1:M
   indices = crossvalind('Kfold',T,K);
   for i=1:K
       test =(indices==i);
       train = ~test;
       dt_test = dt(test,:);
       dt_train = dt(train,:);
       yt_train = yt(train,:);
       yt_test = yt(test,:);
       for jj = 1:tr
            prm.dt = dt_train;
            prm.yt = yt_train;
            prm.tuning = tuningrange(jj);
            prm.usep = 0:pmax;
            res = func1(prm,seed);
            for p = 1:pmax
                [~, idx] = min(res.fcst_error);
                tsr2(i,m,jj,p) = idx;
            end
       end
   end
end

re.tsr2 = reshape(mean(tsr2,[1,2]),[length(tuningrange),param.pmax]);
[i1,i2] = find(re.tsr2==min(re.tsr2,[],'all'));
re.tuning = tuningrange(i1(1));
re.phat = i2(1);

prm = param;
prm.usep = re.phat;
prm.tuning = re.tuning;
res = func1(prm,seed);
res.pmax = prm.usep ;
res.tuning = prm.tuning;
res.tsr2 = re.tsr2;

