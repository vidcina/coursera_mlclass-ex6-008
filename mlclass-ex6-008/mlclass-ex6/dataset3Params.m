function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% 0.01 0.03 0.1 0.3 1 3 10 30

values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
n = length(values);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

errors = zeros(n, n);

for c = 1:n
    for s = 1:n
        C = values(c);
        sigma = values(s);
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        predictions = svmPredict(model, Xval);
        error_value = mean(double(predictions ~= yval));
        errors(c,s) = error_value;
    end
end

[C_index,sigma_index] = find(errors == min(min(errors)))

C = values(C_index);
sigma = values(sigma_index);


% =========================================================================

end
