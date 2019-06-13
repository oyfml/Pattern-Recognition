%%Run this script for LDA part of assignment%%
clc;clear;close all;
fprintf("Loading... Please wait\n");

%Extract training & test images randomly selected (earlier in PCA)
load('data4others.mat');

%Apply LDA to obtain projection matrix W
[W, reorder_idx, class_idx] = LDA_pre(vect_M, labels);

%Project onto w to reduce dimensionality to 2
[y_2dim] = LDA_proj(W, vect_M, 2);
%Project onto w to reduce dimensionality to 3
[y_3dim] = LDA_proj(W, vect_M, 3);
%Project onto w to reduce dimensionality to 9
[y_9dim] = LDA_proj(W, vect_M, 9);
%Project onto w to reduce dimensionality to 20
[y_20dim] = LDA_proj(W, vect_M, 20);
%Project onto w to reduce dimensionality to 21
[y_21dim] = LDA_proj(W, vect_M, 21);

%Sort to class labels for easy colour plotting
y_2d_sort = y_2dim;
temp = y_2d_sort(1,:);
y_2d_sort(1,:) = temp(reorder_idx);
temp = y_2d_sort(2,:);
y_2d_sort(2,:) = temp(reorder_idx);

y_3d_sort = y_3dim;
temp = y_3d_sort(1,:);
y_3d_sort(1,:) = temp(reorder_idx);
temp = y_3d_sort(2,:);
y_3d_sort(2,:) = temp(reorder_idx);
temp = y_3d_sort(3,:);
y_3d_sort(3,:) = temp(reorder_idx);

%Visualise 2D projected data on 2D scatter plot
figure(1);
title('LDA - Visualize 2D projected data vector in 2D space');
%All other faces
scatter(y_2d_sort(1,1:493),y_2d_sort(2,1:493));
hold on;
%My face
scatter(y_2d_sort(1,494:500),y_2d_sort(2,494:500),'filled');

figure(2);
title('LDA - Visualize 2D projected data vector in 2D space');
for i = 1:length(class_idx)-1
    scatter(y_2d_sort(1,class_idx(i):class_idx(i+1)-1),y_2d_sort(2,class_idx(i):class_idx(i+1)-1),'filled');
    hold on;
end

%Visualise 3D projected data on 3D scatter plot
figure(3);
title('LDA - Visualize 3D projected data vector in 3D space');
%All other faces
scatter3(y_3d_sort(1,1:493),y_3d_sort(2,1:493),y_3d_sort(3,1:493));
hold on;
%My face
scatter3(y_3d_sort(1,494:500),y_3d_sort(2,494:500),y_3d_sort(3,494:500),'filled');

figure(4);
title('LDA - Visualize 3D projected data vector in 3D space');
for i = 1:length(class_idx)-1
    scatter3(y_3d_sort(1,class_idx(i):class_idx(i+1)-1),y_3d_sort(2,class_idx(i):class_idx(i+1)-1),y_3d_sort(3,class_idx(i):class_idx(i+1)-1),'filled');
    hold on;
end

%Reconstruct first 3 fisher faces
% figure(5);
% subplot(2, 2, 1);
% imshow(vec2mat(W(:,1), 32),[]);
% title('Fisher face 1');
% subplot(2, 2, 2);
% imshow(vec2mat(W(:,2), 32),[]);
% title('Fisher face 2');
% subplot(2, 2, 3);
% imshow(vec2mat(W(:,3), 32),[]);
% title('Fisher face 3');

%Classify test images with 2 dim, using Nearest Neighbour
[guess_class_2] = NN_classifier(test_M, y_2dim, labels, W(:,1:2));
acc_2 = calculate_err(guess_class_2(1:211), true_class(1:211));
me_acc_2 = calculate_err(guess_class_2(212:214), true_class(212:214));
%Classify test images with 3 dim, using Nearest Neighbour
[guess_class_3] = NN_classifier(test_M, y_3dim, labels, W(:,1:3));
acc_3 = calculate_err(guess_class_3(1:211), true_class(1:211));
me_acc_3 = calculate_err(guess_class_3(212:214), true_class(212:214));
%Classify test images with 9 dim, using Nearest Neighbour
[guess_class_9] = NN_classifier(test_M, y_9dim, labels, W(:,1:9));
acc_9 = calculate_err(guess_class_9(1:211), true_class(1:211));
me_acc_9 = calculate_err(guess_class_9(212:214), true_class(212:214));
%Classify test images with 20 dim, using Nearest Neighbour
[guess_class_20] = NN_classifier(test_M, y_20dim, labels, W(:,1:20));
acc_20 = calculate_err(guess_class_20(1:211), true_class(1:211));
me_acc_20 = calculate_err(guess_class_20(212:214), true_class(212:214));
%Classify test images with 21 dim, using Nearest Neighbour
[guess_class_21] = NN_classifier(test_M, y_21dim, labels, W(:,1:21));
acc_21 = calculate_err(guess_class_21(1:211), true_class(1:211));
me_acc_21 = calculate_err(guess_class_21(212:214), true_class(212:214));

%Display accuracy percentage
fprintf("Accuracy results of CMU data for dimensions 2, 3, 9, 20, 21 is:\n");
fprintf("%2.2f%%   %2.2f%%   %2.2f%%   %2.2f%%   %2.2f%%\n", acc_2*100, acc_3*100, acc_9*100, acc_20*100, acc_21*100);
fprintf("Accuracy results of self-generated data for dimensions 2, 3, 9, 20, 21 is:\n"); 
fprintf("%2.2f%%   %2.2f%%   %2.2f%%   %2.2f%%   %2.2f%%\n", me_acc_2*100, me_acc_3*100, me_acc_9*100, me_acc_20*100, me_acc_21*100);

