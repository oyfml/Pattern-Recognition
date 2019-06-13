function [Training_Err, Test_Err] =  my_Naive_Bayes(Xtrain, Xtest, ytrain, ytest, alpha)
%Performs Beta-Binomial Naive Bayes Classification & Outputs error rate of training &
%test data

%Uses class label prior ML estimate of lambda &
%posterior predictive likelihood with Beta(alpha,alpha) prior

%%Set Class label prior
%Maximum Likelihood estimate of lambda is N1/N,where N1 is no. of data labelled as class 1
class_y1 = find(ytrain == 1);       %Search through test set class labels and sieve locations of class 1
class_y0 = find(ytrain == 0);       %Search through test set class labels and sieve locations of class 0
sum_y1 = length(class_y1);          %Retrieve total no. of all class 1
sum_y0 = length(class_y0);          %Retrieve total no. of all class 0
lambda_ML = sum_y1/(sum_y0+sum_y1); %P(^y = 1 |lambda_ML)= lambda_ML; prior probability of class label 1

%%Set Posterior Predictive Likelihood
%Class 1
N1_c1_train = zeros(size(Xtrain,2),1);     %Vector containing N1 for each feature;N1 is no. of features = 1 for class 1
N0_c1_train = zeros(size(Xtrain,2),1);     %Vector containing N0 for each feature;N0 is no. of features = 0 for class 1
%Class 0    
N1_c0_train = zeros(size(Xtrain,2),1);     %Vector containing N1 for each feature;N1 is no. of features = 1 for class 0
N0_c0_train = zeros(size(Xtrain,2),1);     %Vector containing N0 for each feature;N0 is no. of features = 0 for class 0
for i=1:size(Xtrain,2)                     %Go thru each feature column (1-57) 
    %Class 1
    N1_c1_train(i) = sum(Xtrain(class_y1,i));%Sum of all x=1 for each feature classified as class 1
    N0_c1_train(i) = sum_y1 - N1_c1_train(i);%Sum of all x=0 for each feature classified as class 1
    %Class 0
    N1_c0_train(i) = sum(Xtrain(class_y0,i));%Sum of all x=1 for each feature classified as class 0
    N0_c0_train(i) = sum_y0 - N1_c0_train(i);%Sum of all x=0 for each features classified as class 0
end

%Compute Naive Bayes classfier for Training set
%Input Probability into Training Features
p_train_c1 = zeros(size(Xtrain)); %Probability(likelihood) matrix for class 1
p_train_c0 = zeros(size(Xtrain)); %Probability(likelihood) matrix for class 0
for i = 1:size(Xtrain,1)
    for j = 1:size(Xtrain,2)
        if (Xtrain(i,j)==1)  %Feature value = 1
            %Class 1
            p_train_c1(i,j)=(N1_c1_train(j)+alpha)/(sum_y1+2*alpha);
            %Class 0
            p_train_c0(i,j)=(N1_c0_train(j)+alpha)/(sum_y0+2*alpha);
        else                 %Feature value = 0
            %Class 1
            p_train_c1(i,j)=(N0_c1_train(j)+alpha)/(sum_y1+2*alpha);
            %Class 0
            p_train_c0(i,j)=(N0_c0_train(j)+alpha)/(sum_y0+2*alpha);
        end
    end
end

p_train_c1 = sum(log(p_train_c1'));     %Sum up log probability of likelihood matrix for class 1
p_train_c0 = sum(log(p_train_c0'));     %Sum up log probability of likelihood matrix for class 0
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

%Compute Naive Bayes classfier for Test set
%Input Probability into Training Features
p_test_c1 = zeros(size(Xtest)); %Probability(likelihood) matrix for class 1
p_test_c0 = zeros(size(Xtest)); %Probability(likelihood) matrix for class 0
for i = 1:size(Xtest,1)
    for j = 1:size(Xtest,2)
        if (Xtest(i,j)==1)  %Feature value = 1
            %Class 1
            p_test_c1(i,j)=(N1_c1_train(j)+alpha)/(sum_y1+2*alpha);
            %Class 0
            p_test_c0(i,j)=(N1_c0_train(j)+alpha)/(sum_y0+2*alpha);
        else                 %Feature value = 0
            %Class 1
            p_test_c1(i,j)=(N0_c1_train(j)+alpha)/(sum_y1+2*alpha);
            %Class 0
            p_test_c0(i,j)=(N0_c0_train(j)+alpha)/(sum_y0+2*alpha);
        end
    end
end

p_test_c1 = sum(log(p_test_c1'));     %Sum up log probability of likelihood matrix for class 1
p_test_c0 = sum(log(p_test_c0'));     %Sum up log probability of likelihood matrix for class 0
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

