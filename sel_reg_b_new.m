function [idx,fitsel, beta]=sel_reg_b_new(y,X,N,T)

fitall=zeros(T,N);
sse=zeros(N,1);
betas=zeros(N,1);
for i=1:N
   Xr=X(:,i);
   validRows = Xr ~= 0;  
   X_filt = Xr(validRows); 
   y_filt = y(validRows); 
   b=X_filt\y_filt;
   betas(i,1)=b;
   fit=X_filt*b;
   e=y_filt-fit;
   fitall_p=double(validRows);
   fitall_p(fitall_p == 1) = fit;
   fitall(:,i)=fitall_p;
   sse(i,1)=e'*e;
   %eall(:,i)=e;
end
sse_nonzero = sse(sse ~= 0);     
[minValue, minIndex] = min(sse_nonzero); 
idx = find(sse == minValue, 1); 

%u=eall(:,in);
fitsel=fitall(:,idx);
beta=betas(idx);
end