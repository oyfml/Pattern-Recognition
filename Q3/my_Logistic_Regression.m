function [ Training_Err, Test_Err ] = my_Logistic_Regression( Xtrain, Xtest, ytrain, ytest, lambda )

[N, D]=size(Xtrain);
[N_test]=length(Xtest);

%Estimate w hat
w = zeros(1,D+1);

%Introduce bias term to start of every feature vector
X = [ones(N,1) Xtrain];
X_test = [ones(N_test,1) Xtest];
y = ytrain;

%Begin iteration count
i=1;

%Iteration Loop
while 1
    u = ((1 + exp(-(w * X'))).^(-1))'; %sigmoid function of w transpose x; Nx1 vector
    %Gradient
    g = X'*(u-y) + lambda * [0, w(2:end)]';%D+1x1 vector
    %Hessian
    S = eye(size(u,1)) .* (u * (1-u)');%NxN diagonal matrix
    I = zeros(D+1);
    I(2:end,2:end) = eye(D);%Identity matrix with first row and column zero; D+1xD+1 matrix
    H = X'*S*X + lambda * I;%D+1xD+1 matrix
    %Direction
    d = - inv(H)*g; %D+1x1 vector
    %Note: step size not include, because step size=1
        
    %Compute next w vector
    w_next = w + d';

    %Compare Negative Log likelihood for each w vector until convergence
    NLL = Cost_Function( X, y, w, lambda );
    NLL_next = Cost_Function(X, y, w_next, lambda);
    convergence = abs(NLL - NLL_next)/NLL_next; %use fractional changes
    
    fprintf('%d.  %%diff.:%f  NLL(w):%f   NLL(w+1):%f\n',i, convergence, NLL, NLL_next);

    if convergence < 1e-5
        break;
    else
    %Repeat loop; take next step
    w = w_next;
    i=i+1;
    end
end


% %Compare posterior for class 1 & 0
% %Training set
% py1_train = ((1 + exp(-(w * X'))).^(-1))';
% py0_train = 1-py1_train;
% %Test set
% py1_test = ((1 + exp(-(w * X_test'))).^(-1))';
% py0_test = 1-py1_test;
% %Classification
% Output_Class_train(py1_train>py0_train) = 1;
% Output_Class_train=Output_Class_train';
% Output_Class_test(py1_test>py0_test) = 1;
% Output_Class_test=Output_Class_test';

%Compute Classification using log odds
    Output_Class_train = X * w'; %log odds prediction, N-by-1
    Output_Class_train(Output_Class_train > 0) = 1;
    Output_Class_train(Output_Class_train < 0) = 0;

    Output_Class_test = X_test * w'; %log odds prediction, N-by-1
    Output_Class_test(Output_Class_test > 0) = 1;
    Output_Class_test(Output_Class_test < 0) = 0;
    
%Error Calculation
%Test Error Percentage
Test_Err = sum((sum(Output_Class_test ~= ytest)))/(length(ytest)); %Compares result vs true class, sum differences into percentage        
%Training Error Percentage
Training_Err = sum((sum(Output_Class_train ~= ytrain)))/(length(ytrain));

%End of function

end

