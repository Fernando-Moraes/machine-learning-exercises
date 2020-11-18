function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

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

c_sig = [0.01 0.03 0.1 0.3 1 3 10 30]';
result = zeros(length(c_sig)+length(c_sig),3);
pred_errror = zeros(length(c_sig), length(c_sig));
err=1;

for i = 1 : length(c_sig),
    for j = 1 : length(c_sig),
        c = c_sig(i)
        sig = c_sig(j)
        
        [model] = svmTrain(X,y,c, @(x1, x2) gaussianKernel(x1, x2, sig));
        predictions = svmPredict(model, Xval);
        pred_errror(i,j) = mean(double(predictions ~= yval));
        result(err,:) = [pred_errror(i,j), c, sig];
        err = err+1;
    end;
end;

  sorted_result = sortrows(result, 1);
  C = sorted_result(1,2);
  sigma = sorted_result(1,3);

% =========================================================================

end
