function [ Training_Err, Test_Err ] = my_Gaussian_Naive_Bayes( Xtrain, Xtest, ytrain, ytest )
%Performs Gaussian Naive Bayes Classification & Outputs error rate of 
%training & test data

%Uses class label prior ML estimate of lambda & ML estimate of
%feature likelihood conditioned on class, mean and variance

%Maximum Likelihood Estimate of class prior
%ML estimate of lambda is N1/N,where N1 is no. of data labelled as class 1
class_y1 = find(ytrain == 1);       %Search through test set class labels and sieve locations of class 1
class_y0 = find(ytrain == 0);       %Search through test set class labels and sieve locations of class 0
sum_y1 = length(class_y1);          %Retrieve total no. of all class 1
sum_y0 = length(class_y0);          %Retrieve total no. of all class 0
lambda_ML = sum_y1/(sum_y0+sum_y1); %P(^y = 1 |lambda_ML)= lambda_ML; prior probability of class label 1

%MLE of mean and variance from class 1, feature(i),where i = 1 to 57
% aka MLE of eta jc, where j-feature no., c-class no.
mean_c1 = zeros(size(Xtrain,2),1); 
var_c1 = zeros(size(Xtrain,2),1);
for i=1:size(Xtrain,2)
    mean_c1(i) = sum(Xtrain(class_y1,i))/sum_y1;
    var_c1(i) = sum((Xtrain(class_y1,i) - mean_c1(i)).^2)/sum_y1;
end
%MLE of mean and variance from class 0, feature(i)
mean_c0 = zeros(size(Xtrain,2),1); 
var_c0 = zeros(size(Xtrain,2),1);
for i=1:size(Xtrain,2)
    mean_c0(i) = sum(Xtrain(class_y0,i))/sum_y0;
    var_c0(i) = sum((Xtrain(class_y0,i) - mean_c0(i)).^2)/sum_y0;
end

%For Training Set Classification:
%Compute feature(i) likelihood conditioned on MLE of mean and variance from class 0 & 1
%Substitute mean & variance(from each feature,class)into Univariate
%Gaussian distribution equation; p(x|y=0or1,mew,variance)
%Class 1
p_train_c1 = zeros(size(Xtrain));
for i = 1:size(Xtrain,1)
    for j = 1:size(Xtrain,2)
        p_train_c1(i,j) = log((1/sqrt(2*pi*var_c1(j)))*exp(-0.5*( Xtrain(i,j)-mean_c1(j))/var_c1(j)));
    end
end
p_train_c1 = sum(p_train_c1,2); %Sum across rows of log likelihood into single vector

%Class 0
p_train_c0 = zeros(size(Xtrain));
for i = 1:size(Xtrain,1)
    for j = 1:size(Xtrain,2)
        p_train_c0(i,j) = log((1/sqrt(2*pi*var_c0(j)))*exp(-0.5*( Xtrain(i,j)-mean_c0(j))/var_c0(j)));
    end
end
p_train_c0 = sum(p_train_c0,2); %Sum across rows of log likelihood into single vector

class1_train_log = zeros(size(p_train_c1));   
class0_train_log = zeros(size(p_train_c0));
Output_Class_train = zeros(size(ytrain));
for i=1:length(ytrain)
    class1_train_log(i) = log(lambda_ML) + p_train_c1(i);     %Sum up posterior probability for class 1
    class0_train_log(i) = log(1-lambda_ML) + p_train_c0(i);   %Sum up posterior probability for class 0
    if(class1_train_log(i) > class0_train_log(i))
        Output_Class_train(i) = 1;   %Classified as class 1
    else
        Output_Class_train(i) = 0;   %Classified as class 0
    end
end

%For Test Set Classification:
%Compute feature(i) likelihood conditioned on MLE of mean and variance from class 0 & 1
%Substitute mean & variance(from each feature,class)into Univariate
%Gaussian distribution equation; p(x|y=0or1,mew,variance)
%Class 1
p_test_c1 = zeros(size(Xtest));
for i = 1:size(Xtest,1)
    for j = 1:size(Xtest,2)
        p_test_c1(i,j) = log((1/sqrt(2*pi*var_c1(j)))*exp(-0.5*( Xtest(i,j)-mean_c1(j))/var_c1(j)));
    end
end
p_test_c1 = sum(p_test_c1,2); %Sum across rows of log likelihood into single vector

%Class 0
p_test_c0 = zeros(size(Xtest));
for i = 1:size(Xtest,1)
    for j = 1:size(Xtest,2)
        p_test_c0(i,j) = log((1/sqrt(2*pi*var_c0(j)))*exp(-0.5*( Xtest(i,j)-mean_c0(j))/var_c0(j)));
    end
end
p_test_c0 = sum(p_test_c0,2); %Sum across rows of log likelihood into single vector

class1_test_log = zeros(size(p_test_c1));   
class0_test_log = zeros(size(p_test_c0));
Output_Class_test = zeros(size(ytest));
for i=1:length(ytest)
    class1_test_log(i) = log(lambda_ML) + p_test_c1(i);     %Sum up posterior probability for class 1
    class0_test_log(i) = log(1-lambda_ML) + p_test_c0(i);   %Sum up posterior probability for class 0
    if(class1_test_log(i) > class0_test_log(i))
        Output_Class_test(i) = 1;   %Classified as class 1
    else
        Output_Class_test(i) = 0;   %Classified as class 0
    end
end

%Error Calculation
%Test Error Percentage
Test_Err = sum((sum(Output_Class_test ~= ytest)))/(length(ytest)); %Compares result vs true class, sum differences into percentage        
%Training Error Percentage
Training_Err = sum((sum(Output_Class_train ~= ytrain)))/(length(ytrain));

%End of function
end

