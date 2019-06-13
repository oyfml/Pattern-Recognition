function [acc] = calculate_err(guess_class, true_class)
%Calculate error percentage between NN class label vs true class label
%Input: NN derived class labels
%       true class labels
%Output: accuracy %

%Spot difference in class labels
diff = guess_class - true_class;

%Label 1 to location with non-zero diff. value
diff((diff ~= 0)) = 1;

%Sum # of errors
err = sum(sum(diff));

%Find accuracy %
acc = (length(true_class) - err)/length(true_class);

end


