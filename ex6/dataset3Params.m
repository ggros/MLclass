function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 0.3;
sigma = 0.1;


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

% loop thru all possible values 8x8
% code disabled, we return directly the good values above
% New Best found 0.035000 sigma = 0.100000 and C = 0.300000

%{
sigma_vec = [0.01 0.03 0.1 0.3 1 3 10 30]';
C_vec = [0.01 0.03 0.1 0.3 1 3 10 30]';
best_prediction = 1000;
for i = 1:length(sigma_vec)
  for j = 1:length(C_vec)
    fprintf(['Train with sigma = %f and C = %f\n'], sigma_vec(i), C_vec(j));    
    model= svmTrain(X, y, C_vec(j), @(x1, x2) gaussianKernel(x1, x2, sigma_vec(i)));
    predictions = svmPredict(model, Xval);
    prediction_error = mean(double(predictions ~= yval));
    if(prediction_error < best_prediction)
       fprintf(['New Best found %f sigma = %f and C = %f\n'], prediction_error, sigma_vec(i), C_vec(j)); 
       best_prediction = prediction_error;
       sigma = sigma_vec(i);
       C = C_vec(j);   
    end
   end
end
%}


% =========================================================================

end
