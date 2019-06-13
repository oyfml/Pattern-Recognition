function [new_M, EF] = PCA(vect_M,dim)
%Performs PCA on vectorised images in the form of a vectorised matrix
%Each column represent img number, row represents # of dimensions
%Input: vectorised Matrix, desired number of dimensions
%Output: projected vectorised Matrix onto new PCs with reduced dimensions 
%        first dim-th eigenfaces/eigenvectors

%Find the mean vector of the data; size denominator is replacable with 500
mean = sum(vect_M,2) / size(vect_M,2);
mean = repmat(mean,[1,size(vect_M,2)]);

%Centre data origin at mean
mean_M = vect_M - mean;

%Perform SVD of input, s.t. input = U*S*V'
%U - mxm(# of dim), D - mxn, V - nxn(# of data points)
[U,D,V] = svd(mean_M);

%Extract first dim-th eigenvectors with highest eigenvalues
U_new = U(:,1:dim);

%Obtain transformation matrix G
G = U_new;

%Project onto new PCs
new_M = G'*vect_M;

%Eigenfaces
EF = U_new;
end

