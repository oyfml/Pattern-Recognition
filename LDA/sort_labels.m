function [reorder_idx, class_idx] = sort_labels(labels)
%Performs sorting of randomly permuated label list for same class colour
%plot in main function
%Input: label list
%Output: sorted label list's reorder index
%        starting index for new class in sorted label list

%Sort label list
[labels, reorder_idx] = sort(labels);

%Initialise starting class index for 21 img classes
class_idx = 1;
for i = 1:length(labels)-1
    if labels(i) ~= labels(i+1)
        %Add to class index list if new class
        class_idx = cat(2,class_idx,i+1);
    end
end
class_idx = cat(2,class_idx,length(labels)+1);
end

