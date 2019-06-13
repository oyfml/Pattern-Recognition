function [vect_M, label_list] = randomselect(d, img_num)
%Randomly select img_num images from CMU PIE TRAIN/TEST file as training or test set
%Input: folder directory d
%       desired number of selected images         
%Output: matrix of selected images in vectorised form, label list of imgs

%Folder directory
%d = 'PIE/CMU_PIE_TRAIN' OR 'PIE/CMU_PIE_TEST' ;

%Obtain f, structure of filenames in folder
f = dir([d '/*.jpg']);

%Number of img files in folder
n = numel(f);

%Randomly permutate a list of img_num images out of n files
p_list = randperm(n,img_num);

%Create vectorised matrix of image data & its label list
vect_M = zeros(32*32,img_num);
label_list = zeros(1,img_num);
for i = 1:img_num
    %vectorised matrix
    temp = im2double(imread([d,'/',f(p_list(i)).name]));
    temp = temp';
    temp = temp(:); %pixel index go from L->R,T->B of matrix
    vect_M(:,i) = temp;
    %label list
    cell_content = f(p_list(i)).name;
    label_list(1,i) = string(cell_content(1:2));
    %convert first 2 digit (label) from string to number
end

end

