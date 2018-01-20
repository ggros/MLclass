function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% X is the change in water level, with one extra column of 1 (2 x 12)
% theta for linear reg is only 2x1
n = size(theta);
h = X * theta;

J = 1/(2*m) * sum((h - y).^2) + lambda/(2*m) * sum(theta(2:end).^2);


%regularization all but theta 0
reg = [ 0 ; ones(n-1,1)] * lambda / m;
grad = (1/m) * ( X' * ( h - y )) + reg.*theta;


% =========================================================================

grad = grad(:);

end
