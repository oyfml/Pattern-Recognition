function [vect_M, label_list] = retrieve_test()
%Retrieve test images from CMU PIE TEST file as test set
%Input: -NIL-
%Output: matrix of selected images in vectorised form, label list of imgs

%Folder directory
d = 'PIE/CMU_PIE_TEST';

%Obtain f, structure of filenames in folder
f = dir([d '/*.jpg']);

%Number of img files in folder
n = numel(f);

%Create vectorised matrix of image data & its label list
vect_M = zeros(32*32,n);
label_list = zeros(1,n);
for i = 1:n-3
    %vectorised matrix
    temp = im2double(imread([d,'/',f(i).name]));
    temp = temp';
    temp = temp(:); %pixel index go from L->R,T->B of matrix
    vect_M(:,i) = temp;
    %label list
    cell_content = f(i).name;
    label_list(1,i) = string(cell_content(1:2));
    %convert first 2 digit (label) from string to number
end

end

