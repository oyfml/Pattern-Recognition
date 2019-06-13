%Question 1 Beta Binomial Naive Bayes

%This script performs initialisation and data processing before calling
%my_Naive_Bayes for training & test error for different values of alpha

%Input Training Data
clc;
clear;
load('spamData.mat');

%Data Processing: Binarization of training & test sets
Xtrain = Xtrain >0;               
Xtest = Xtest >0;

%Initialising alpha for prior Beta for feature likelihood
alpha = 0:0.5:100;

%Iterative Naive Bayes Classification using different alpha values
Training_Err = zeros(1,length(alpha));
Test_Err = zeros(1,length(alpha));

fprintf('Training Started.\n Please wait...\n');
for i=1:length(alpha)
    [ Training_Err(i), Test_Err(i) ] =  my_Naive_Bayes( Xtrain, Xtest, ytrain, ytest, alpha(i) );
end
fprintf('Classification Complete.\n Plotting data...\n');

%Data Representation
hold on
plot(alpha,Training_Err,'b');
plot(alpha,Test_Err,'r');
grid on;
title('Plot of error against alpha');
xlabel('alpha');
ylabel('Error');
legend('training error','testing error');
hold off;

%Display Training & Test error rates for alpha = 1, 10, 100
disp('For alpha = 1:');
fprintf('Training error is %f, test error is %f.\n',Training_Err(3),Test_Err(3));
disp('For alpha = 10:');
fprintf('Training error is %f, test error is %f.\n',Training_Err(21),Test_Err(21));
disp('For alpha = 100:');
fprintf('Training error is %f, test error is %f.\n',Training_Err(201),Test_Err(201));
%Note: alpha(3)=1,alpha(21)=10,alpha(201)=100

%End of Script

