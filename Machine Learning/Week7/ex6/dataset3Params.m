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
cRange = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigmaRange = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
error_min = inf;
error_temp = 0;
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

for i = 1:size(cRange,1)
    for j = 1:size(sigmaRange,1)
        model= svmTrain(X, y, cRange(i), @(x1, x2) gaussianKernel(x1, x2, sigmaRange(j)));
        predictions = svmPredict(model, Xval);
        error_temp = mean(double(predictions ~= yval));
        fprintf('prediction error: %f\n', error_temp);
        if(error_temp < error_min)
            C = cRange(i);
            sigma = sigmaRange(j);
            error_min = error_temp;
            fprintf('[C, sigma] = [%f %f]\n', C, sigma);
        end
        fprintf('--------\n');
    end    
end

fprintf('\nfinish searching.\nBest value [C, sigma] = [%f %f] with prediction error = %f\n\n', C, sigma, error_min);
% =========================================================================

end
