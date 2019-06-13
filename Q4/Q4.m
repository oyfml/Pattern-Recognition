%Question 4 K-Nearest Neighbors
%This script performs initialisation and data processing before calling
%my_KNN for training & test error

%Input Training Data
clc;
clear;
load('spamData.mat');

%Data Processing: log-transforming training & test sets
Xtrain = log(Xtrain + 0.1);
Xtest = log(Xtest + 0.1);

%Initialising K
K = [(1:10),(15:5:100)];

%Iterative KNN using different K nearest neighbour values
Training_Err = zeros(1,length(K));
Test_Err = zeros(1,length(K));

%Run KNN Classifier
fprintf('Training Started. Please wait...\n');
for i=1:length(K)
    [ Training_Err(i), Test_Err(i) ] = my_KNN( Xtrain, Xtest, ytrain, ytest , K(i));
    fprintf('Training Complete for K = %d.\n',K(i));
end
fprintf('Classification Complete.\n Plotting data...\n');

%Data Representation
hold on
plot(K,Training_Err,'b');
plot(K,Test_Err,'r');
grid on;
title('Plot of error against K');
xlabel('K');
ylabel('Error');
legend('training error','testing error');
hold off;

%Display Training & Test error rates for K = 1, 10, 100
disp('For K = 1:');
fprintf('Training error is %f, test error is %f.\n',Training_Err(1),Test_Err(1));
disp('For K = 10:');
fprintf('Training error is %f, test error is %f.\n',Training_Err(10),Test_Err(10));
disp('For K = 100:');
fprintf('Training error is %f, test error is %f.\n',Training_Err(28),Test_Err(28));
%Note: K(1)=1,K(10)=10,K(28)=100

%End of Script