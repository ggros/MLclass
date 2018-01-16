function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% in the sample size(X) = 5000 400

%recode y(i) to be a vector of size k with a 1 in the position of correct answer
Y = [];
for i = 1:num_labels
  Y = [Y y == i];
endfor
% size(Y) = 5000 10

% ------------------------------
% feed forward to calculate h(x)
% ------------------------------

% Add ones to the X data matrix -> 401 input for activation a1
A1 = [ones(m, 1) X];

Z2 = A1 * Theta1';

% calculate activation a2
A2 = [ones(size(Z2, 1), 1) sigmoid(Z2)];

% calculate Z for layer 3 = output layer
Z3 = A2 * Theta2';

%calculate h(x) for every sample -> matrix H is 5000 10 (h is vector of size k, here k=10)
A3 = sigmoid(Z3);
H = A3;


% ------------------------------
% Calculate cost function J
% ------------------------------

% from lrCostFunction of logistic regression
%J = (1/m).*( -y'*log(h) - (1 - y)'*log(1 - h) )

% loop over samples and sum cost with for loops (v1)
%{
for i = 1:m
    for k = 1:num_labels
      J = J + ( -Y(i,k)*log(H(i,k)) - (1-Y(i,k))*log(1 - H(i,k)) );
    endfor
endfor
J = J/m;
%}

% calculate using element wise product and then sum all elements (to do it in one line)
J = 1/m * sum(sum( -Y.*log(H) - (1-Y).*log(1 - H) ));

%calculate regularization (skip col 1 which is weight for the bias)
reg = lambda/(2*m) * ( sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)) );

% sum(sum( can also be computed using ones()
% http://stattrek.com/matrix-algebra/sum-of-elements.aspx
%{
t1Square = Theta1(:,2:end).^2;
t2Square = Theta2(:,2:end).^2;
reg = lambda/(2*m) * (...
  ones(1, size(t1Square,1)) * t1Square * ones(size(t1Square,2), 1)...
  +ones(1, size(t2Square,1)) * t2Square * ones(size(t2Square,2), 1)...
);
%}

J = J + reg;


% ------------------------------
% Backpropagation
% ------------------------------

% backpropagation using for loop
for i = 1:m
  delta_3 = (A3(i,:) - Y(i,:))'; % transpose to get a vector
  delta_2 = (Theta2'*delta_3)(2:end) .* sigmoidGradient(Z2(i,:)');
  
  Theta2_grad = Theta2_grad + delta_3 * (A2(i,:));
  Theta1_grad = Theta1_grad + delta_2 * (A1(i,:));
  
endfor

Theta2_grad = 1/m * (Theta2_grad + lambda * [zeros(size(Theta2,1),1) Theta2(:,2:end)] );
Theta1_grad = 1/m * (Theta1_grad + lambda * [zeros(size(Theta1,1),1) Theta1(:,2:end)] );


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
