%%%Run this script for PCA part of assignment%%%
clc;clear;close all;
fprintf("Loading... Please wait\n");

%Training set: 500 (70%)
%Randomly select 493 images from CMU PIE TRAIN
[vect_M, labels] = randomselect('PIE/CMU_PIE_TRAIN',493);
%Add 7 more self images into training data
[add, add_labels] = randomselect('PIE/ME_TRAIN',7);
vect_M = [vect_M, add];
labels = [labels, add_labels];%label my images as img_ID 69

%Perform PCA to reduce img vector dim to 2
y_2dim = PCA(vect_M,2);
%Perform PCA to reduce img vector dim to 3
[y_3dim, eig_vec3] = PCA(vect_M,3);

%Sort to class labels for easy colour plotting
[reorder_idx, class_idx] = sort_labels(labels);
temp1 = y_2dim(1,:);
temp2 = y_2dim(2,:);
y_2dim(1,:) = temp1(reorder_idx);
y_2dim(2,:) = temp2(reorder_idx);

temp1 = y_3dim(1,:);
temp2 = y_3dim(2,:);
temp3 = y_3dim(3,:);
y_3dim(1,:) = temp1(reorder_idx);
y_3dim(2,:) = temp2(reorder_idx);
y_3dim(3,:) = temp2(reorder_idx);


%Plot scatter plot, all colour & only me img highlighted
%Visualise 2D projected data on 2D scatter plot
figure(1);
title('PCA - Visualize 2D projected data vector in 2D space');
%All other faces
scatter(y_2dim(1,1:493),y_2dim(2,1:493));
hold on;
%My face
scatter(y_2dim(1,494:500),y_2dim(2,494:500),'filled');

figure(2);
title('PCA - Visualize 2D projected data vector in 2D space');
for i = 1:length(class_idx)-1-1
    scatter(y_2dim(1,class_idx(i):class_idx(i+1)-1),y_2dim(2,class_idx(i):class_idx(i+1)-1),'filled');
    hold on;
end

%Visualise 3D projected data on 3D scatter plot
figure(3);
title('PCA - Visualize 3D projected data vector in 3D space');
%All other faces
scatter3(y_3dim(1,1:493),y_3dim(2,1:493),y_3dim(3,1:493));
hold on;
%My face
scatter3(y_3dim(1,494:500),y_3dim(2,494:500),y_3dim(3,494:500),'filled');

figure(4);
title('PCA - Visualize 3D projected data vector in 3D space');
for i = 1:length(class_idx)-1-1
    scatter3(y_3dim(1,class_idx(i):class_idx(i+1)-1),y_3dim(2,class_idx(i):class_idx(i+1)-1),y_3dim(3,class_idx(i):class_idx(i+1)-1),'filled');
    hold on;
end

%Reconstruct first 3 eigenfaces
figure(5);
subplot(2, 2, 1);
imshow(vec2mat(eig_vec3(:,1), 32),[]);
title('Eigenface 1');
subplot(2, 2, 2);
imshow(vec2mat(eig_vec3(:,2), 32),[]);
title('Eigenface 2');
subplot(2, 2, 3);
imshow(vec2mat(eig_vec3(:,3), 32),[]);
title('Eigenface 3');

%Perform PCA to reduce dim to 40
[y_40dim, eig_vec40] = PCA(vect_M,40);
%Perform PCA to reduce dim to 80
[y_80dim, eig_vec80] = PCA(vect_M,80);
%Perform PCA to reduce dim to 200
[y_200dim, eig_vec200] = PCA(vect_M,200);

%Test set: 214 (30%)
%Randomly select 211 images from CMU PIE TEST
[test_M, true_class] = randomselect('PIE/CMU_PIE_TEST',211);
%Add 3 more self images to test data
[add_test,add_class] = randomselect('PIE/ME_TEST',3);
test_M = [test_M, add_test];
true_class = [true_class, add_class];

%Classify dim 40 test images according to nearest neighbour
[guess_class_40] = NN_classifier(test_M, y_40dim, labels, eig_vec40);
acc_40 = calculate_err(guess_class_40(1:211), true_class(1:211));
me_acc_40 = calculate_err(guess_class_40(212:214), true_class(212:214));
%Classify dim 80 test images according to nearest neighbour
[guess_class_80, ytest_80] = NN_classifier(test_M, y_80dim, labels, eig_vec80);
acc_80 = calculate_err(guess_class_80(1:211), true_class(1:211));
me_acc_80 = calculate_err(guess_class_80(212:214), true_class(212:214));
%Classify dim 200 test images according to nearest neighbour
[guess_class_200, ytest_200] = NN_classifier(test_M, y_200dim, labels, eig_vec200);
acc_200 = calculate_err(guess_class_200(1:211), true_class(1:211));
me_acc_200 = calculate_err(guess_class_200(212:214), true_class(212:214));

%Display accuracy percentages
fprintf("Accuracy results of CMU data for dimensions 40, 80, 200 is:\n");
fprintf("%2.2f%%   %2.2f%%   %2.2f%%\n", acc_40*100, acc_80*100, acc_200*100);
fprintf("Accuracy results of self-generated data for dimensions 40, 80, 200 is:\n"); 
fprintf("%2.2f%%   %2.2f%%   %2.2f%%\n", me_acc_40*100, me_acc_80*100, me_acc_200*100);

%Save variables to workspace for other folders LDA, SVM
%save('data4others','vect_M','labels','y_80dim','y_200dim');
PATH = pwd;
P_LDA = strrep(PATH,'PCA','LDA');
P_LDA = [P_LDA '/data4others.mat'];
save(P_LDA,'vect_M','labels','test_M','true_class');
P_SVM = strrep(PATH,'PCA','SVM');
P_SVM = [P_SVM '/data4others.mat'];
save(P_SVM,'labels','y_80dim','y_200dim','true_class','ytest_80','ytest_200');