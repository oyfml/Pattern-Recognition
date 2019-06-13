%Question 3 Logistic Regression
%This script performs initialisation and data processing before calling
%my_Logistic_Regression for training & test error

%Input Training Data
clc;
clear;
load('spamData.mat');

%Data Processing: log-transforming training & test sets
Xtrain = log(Xtrain + 0.1);
Xtest = log(Xtest + 0.1);

%Initialising regularisation parameter lambda
lambda = [1:10,15:5:100];

%Iterative Logistic Regression Newtons Method using different lambda values
Training_Err = zeros(1,length(lambda));
Test_Err = zeros(1,length(lambda));

%Run Logistic Regression
fprintf('Training Started.\n Please wait...\n');
for i=1:length(lambda)
    [ Training_Err(i), Test_Err(i) ] =  my_Logistic_Regression( Xtrain, Xtest, ytrain, ytest, lambda(i) );
    fprintf('Training Complete for lambda = %d.\n',lambda(i));
end
fprintf('Classification Complete.\n Plotting data...\n');

%Data Representation
hold on
plot(lambda,Training_Err,'b');
plot(lambda,Test_Err,'r');
grid on;
title('Plot of error against lambda');
xlabel('lambda');
ylabel('Error');
legend('training error','testing error');
hold off;

%Display Training & Test error rates for lambda = 1, 10, 100
disp('For lambda = 1:');
fprintf('Training error is %f, test error is %f.\n',Training_Err(1),Test_Err(1));
disp('For lambda = 10:');
fprintf('Training error is %f, test error is %f.\n',Training_Err(10),Test_Err(10));
disp('For lambda = 100:');
fprintf('Training error is %f, test error is %f.\n',Training_Err(28),Test_Err(28));
%Note: lambda(1)=1,lambda(10)=10,lambda(28)=100

%End of Script

