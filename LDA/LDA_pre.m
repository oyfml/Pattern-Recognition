function [W, reorder_idx, class_idx] = LDA_pre(vect_M, labels)
%Performs pre-processing of LDA before projection
%Input: vectorised training image in matrix form
%       class labels for corresponding image vector       
%Output: projection matrix
%        sorted label list's reorder index
%        starting index for new class in sorted label list

%Sort according to classes
[reorder_idx, class_idx] = sort_labels(labels);
for i = 1:size(vect_M,1)
    temp = vect_M(i,:);
    vect_M(i,:) = temp(reorder_idx);
end

%No. of classes
numclass = length(class_idx) - 1;

%Class specific mean vector; each column represents mean vector of a class(in ascending order)
CSMV = zeros(size(vect_M,1),numclass); %21 classes
%Class specific covariance matrix; every 3rd dim represents one class
CSCM = zeros(size(vect_M,1),size(vect_M,1),numclass);
%Number of img data per class, ni
numperclass = zeros(1,numclass);
for i = 1:numclass
    numperclass(i) = class_idx(i+1) - class_idx(i);
    
    CSMV(:,i) = sum(vect_M(:,class_idx(i):class_idx(i+1)-1),2);
    CSMV(:,i) = CSMV(:,i)/numperclass(i);
    
    for j = 0:numperclass(i)-1
        CSCM(:,:,i) = CSCM(:,:,i) + (vect_M(:,(class_idx(i)+j)) - CSMV(:,i)) ...
        *(vect_M(:,(class_idx(i)+j)) - CSMV(:,i))';
    end
    CSCM(:,:,i) = CSCM(:,:,i)/numperclass(i);
end

%Total mean vector
N = sum(numperclass);
TMV = sum(vect_M,2)/N;

%Within class scatter
SW = zeros(size(vect_M,1));
%Between class scatter
SB = zeros(size(vect_M,1));
for i = 1:numclass
    
    SW = SW + CSCM(:,:,i) * numperclass(i)/N;

    SB = SB + (CSMV(:,i) - TMV)*(CSMV(:,i) - TMV)' * numperclass(i)/N;
    
end

%Compute eigenvectors & sort starting from highest eigenvalue
%Largest eigenvalue = max J(W)
eigen_M = pinv(SW) * SB;
[eig_vec, eig_val] = eig(eigen_M);
eig_val = diag(eig_val);
[eig_val, sort_idx] = sort(real(eig_val),'descend');
eig_vec = real(eig_vec);
for i = 1:length(eig_val)
    temp = eig_vec(1,:);
    eig_vec(1,:) = temp(sort_idx);
end

%Obtain projection matrix W*
W = eig_vec;

end

