%%%Run this script for SWM part of assignment%%%
clc;clear;close all;
fprintf("Loading... Please wait\n");
load('data4others.mat')

%Formatting matrix for correct input into mex file
labels = labels';
y_80dim = y_80dim';
y_200dim = y_200dim';
true_class = true_class';
ytest_80 = ytest_80';
ytest_200 = ytest_200';


%Call SVM functions (include mex file in current folder)

%Dimensionality 80
%Linear ,Penalty C = 0.01
model_80_1 = svmtrain(labels, y_80dim, '-t 0 -c 0.01');
[guess_label_80_1] = svmpredict(true_class, ytest_80, model_80_1);
%Linear ,Penalty C = 0.1
model_80_2 = svmtrain(labels, y_80dim, '-t 0 -c 0.1');
[guess_label_80_2] = svmpredict(true_class, ytest_80, model_80_2);
%Linear ,Penalty C = 1
model_80_3 = svmtrain(labels, y_80dim, '-t 0 -c 1');
[guess_label_80_3] = svmpredict(true_class, ytest_80, model_80_3);

%Dimensionality 200
%Linear ,Penalty C = 0.01
model_200_1 = svmtrain(labels, y_200dim, '-t 0 -c 0.01');
[guess_label_200_1] = svmpredict(true_class, ytest_200, model_200_1);
%Linear ,Penalty C = 0.1
model_200_2 = svmtrain(labels, y_200dim, '-t 0 -c 0.1');
[guess_label_200_2] = svmpredict(true_class, ytest_200, model_200_2);
%Linear ,Penalty C = 1
model_200_3 = svmtrain(labels, y_200dim, '-t 0 -c 1');
[guess_label_200_3] = svmpredict(true_class, ytest_200, model_200_3);

%Calculate accuracy
acc_80_1 = calculate_err(guess_label_80_1, true_class);
acc_80_2 = calculate_err(guess_label_80_2, true_class);
acc_80_3 = calculate_err(guess_label_80_3, true_class);
acc_200_1 = calculate_err(guess_label_200_1, true_class);
acc_200_2 = calculate_err(guess_label_200_2, true_class);
acc_200_3 = calculate_err(guess_label_200_3, true_class);

%Display accuracy percentages
fprintf("Accuracy results of CMU data for dimensions 80 is:\n");
fprintf("c = 0.01 => %2.2f%% | c = 0.1 => %2.2f%% | c = 1 => %2.2f%%\n", acc_80_1*100, acc_80_2*100, acc_80_3*100);
fprintf("Accuracy results of CMU data for dimensions 200 is:\n"); 
fprintf("c = 0.01 => %2.2f%% | c = 0.1 => %2.2f%% | c = 1 => %2.2f%%\n", acc_200_1*100, acc_200_2*100, acc_200_3*100);
