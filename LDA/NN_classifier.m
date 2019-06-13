function [guess_class] = NN_classifier(test_M, y_train, train_label, eigvec)
%Performs Nearest Neighbour classification (1-NN) on test images
%Input: vectorised test datas in matrix form
%       projected vectorised training data matrix with reduced dimensions
%       class label of training data
%       eigenvectors of reduced dim
%Output: class label of test data using rule of NN 

%Project test data onto new PCs
y_test = eigvec'*test_M;

%Perform 1-NN calculation using Euclidean distance for each test data
guess_class = zeros(1,size(y_test,2));
for i = 1: size(y_test,2)
    y_temp = repmat(y_test(:,i), [1, size(y_train,2)]);
    dist_m = y_train - y_temp;
    [~, idx]= min(sum(dist_m.^2));
    guess_class(1,i) = train_label(1,idx);

end


end

