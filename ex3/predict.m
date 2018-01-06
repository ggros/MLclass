function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% hidden layer has 25 nodes
% size(Theta1) = 25   401 
% size(X) =  [5000 400]

% Add ones to the X data matrix
X = [ones(m, 1) X];

% layer1 25 5000
layer1 = sigmoid(X * Theta1');

% Add ones in first column to the layer1 data matrix
layer1 = [ones(size(layer1, 1), 1) layer1];
% size layer1 = 5000 26

% size(Theta2) = 10   26
[max, p] = max(sigmoid(layer1 * Theta2'), [], 2);







% =========================================================================


end
