function [y ] = mtrans( x, tc )
% this function transforms the individual collumns in x based on the transformation code in tc
% INPUTS:
%       x  is T x k data matriz
%       tc is 1 x k vector of transformation codes
% OUTPUTS:
%       y is T-2 x k data matrix (containing periods 3 to T. First two periods are always dropped even if no transformation is performed)
%
% codes:
%        1 for no transformation
%        2 for first-difference transformation
%        3 for second differences
%        4 for logs
%        5 for first-differenced logs
%        6 for second-differenced logs

[T,k]=size(x); % get the dimensions
y=NaN(T-2,k); % initialize

for i=1:k
    
    if tc(i)==1
       y(:,i)=x(3:T,i); 
    end
    if tc(i)==2
       y(:,i)=x(3:T,i)-x(2:T-1,i); 
    end
    if tc(i)==3
       y(:,i)=(x(3:T,i)-x(2:T-1,i)) - ( x(2:T-1,i)-x(1:T-2,i) ); 
    end    
    if tc(i)==4
       y(:,i)=log(x(3:T,i)); 
    end    
    if tc(i)==5
       y(:,i)=log(x(3:T,i))-log(x(2:T-1,i)); 
    end 
    if tc(i)==6
       y(:,i)=(log(x(3:T,i))-log(x(2:T-1,i))) - ( log(x(2:T-1,i))-log(x(1:T-2,i)) ); 
    end     
end
 