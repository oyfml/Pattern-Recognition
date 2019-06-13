%Question 2 Gaussian Naive Bayes
%This script performs initialisation and data processing before calling
%my_Gaussian_Naive_Bayes for training & test error

%Input Training Data
clc;
clear;
load('spamData.mat');

%Data Processing: log-transforming training & test sets
Xtrain = log(Xtrain + 0.1);
Xtest = log(Xtest + 0.1);

%Run Gaussian Naive Bayes Classifier
fprintf('Training Started.\n ');

[ Training_Err, Test_Err ] = my_Gaussian_Naive_Bayes( Xtrain, Xtest, ytrain, ytest );

fprintf('Classification Complete.\n ');

%Display Training & Test error rate
fprintf('Training error is %f, test error is %f.\n',Training_Err,Test_Err);

%End of Script

