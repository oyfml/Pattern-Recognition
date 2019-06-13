function [ NLL ] = Cost_Function( X, y, w, lambda )
%Computes cost function equation NLL(negative Log Likelihood)
 
    py_1 = ((1 + exp(-(w * X'))).^(-1))';%Sigmoid
    py_0 = 1 - py_1;
    Likelihood = py_1 .* y + py_0 .* (1-y); 
    NLL = -sum(log(Likelihood));
 
    %l2 regularisation
    NLL = NLL + lambda * (w(2:end)*w(2:end)'); 
    
end

