function [ Training_Err, Test_Err ] = my_KNN( Xtrain, Xtest, ytrain, ytest , K )
%Performs KNN Classification on training and test set using Euclidean dist.

[N] = length(Xtrain);
[N_test]=length(Xtest);

%KNN on Training Set given Training Set
%Create NxN matrix of Euclidean Dist between new data and training data, 
%where the row no.(i), column no.(j) element is the
%j-th row of new Dx1-feature vector(Xtrain) compared with i-th row Dx1-feature vector in Xtrain
Euclidean_Dist = zeros(N,N);
for j=1:N
    Euclidean_Dist(:,j) = (sum((abs(repmat(Xtrain(j,:),N,1) - Xtrain)).^2,2)).^(1/2); %Euclidean dist formula
end

%Extract nearest K neighbours using Euclidean dist.
K_min = sort(Euclidean_Dist);%Sort each column from min to max
K_min = K_min(K,:);
%Extract nearest K neighbour location & identify class label for each neighbour
y_id = zeros(K,N);
neighbour_counter=0;%counts no. of neighbours from 1-K
for j=1:N
    for i=1:N
        if((neighbour_counter<K)&&(Euclidean_Dist(i,j)<=K_min(j)))
            neighbour_counter=neighbour_counter+1;
            y_id(neighbour_counter,j)=ytrain(i);
        end
    end
    neighbour_counter=0;%reset counter
end

%Compute posterior of training set
%Class 1
py1_train = sum(y_id)/K;
%Class 0
py0_train = 1 - py1_train;

%Classification of training set
Output_Class_train=zeros(1,N);
Output_Class_train(py1_train>py0_train) = 1;
Output_Class_train=Output_Class_train';

%KNN on Test Set given Training Set
%Create NxN_test matrix of Euclidean Dist between new data and training data, 
%where the row no.(i), column no.(j) element is the
%j-th row of new Dx1-feature vector(Xtest) compared with i-th row Dx1-feature vector in Xtrain
Euclidean_Dist = zeros(N,N_test);
for j=1:N_test
    Euclidean_Dist(:,j) = (sum((abs(repmat(Xtest(j,:),N,1) - Xtrain)).^2,2)).^(1/2); %Euclidean dist formula
end

%Extract nearest K neighbours using Euclidean dist.
K_min = sort(Euclidean_Dist);%Sort each column from min to max
K_min = K_min(K,:);
%Extract nearest K neighbour location & identify class label for each neighbour
y_id_test = zeros(K,N_test);
neighbour_counter=0;%counts no. of neighbours from 1-K
for j=1:N_test
    for i=1:N
        if((neighbour_counter<K)&&(Euclidean_Dist(i,j)<=K_min(j)))
            neighbour_counter=neighbour_counter+1;
            y_id_test(neighbour_counter,j)=ytrain(i);
        end
    end
    neighbour_counter=0;%reset counter
end

%Compute posterior for test set
%Class 1
py1_test = sum(y_id_test)/K;
%Class 0
py0_test = 1 - py1_test;

%Classification of test set
Output_Class_test=zeros(1,N_test);
Output_Class_test(py1_test>py0_test) = 1;
Output_Class_test=Output_Class_test';

%Error Calculation
%Test Error Percentage
Test_Err = sum((sum(Output_Class_test ~= ytest)))/(length(ytest)); %Compares result vs true class, sum differences into percentage        
%Training Error Percentage
Training_Err = sum((sum(Output_Class_train ~= ytrain)))/(length(ytrain));

end
%End of function
