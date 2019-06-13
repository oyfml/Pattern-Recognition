function [y] = LDA_proj(W, vect_M, d)
%Performs LDA projection to reduce dimensionality
%Input: projection matrix
%       vectorised img matrix
%       reduced dimensionality
%Output:

y = W(:,1:d)'*vect_M;

end

