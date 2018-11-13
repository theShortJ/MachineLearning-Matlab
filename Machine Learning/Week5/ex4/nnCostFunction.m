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

Xt = [ones(m,1) X];
eye_matrix = eye(num_labels);
y_matrix = eye_matrix(y,:);

fprintf('\n Size X \n'); size(Xt)
fprintf('\n Size theta1 and theta2 \n'); size(Theta1); size(Theta2)

    z2 = Xt*Theta1';
    a2 = sigmoid(z2);
    a2 = [ones(size(a2,1),1) a2];
    z3 = a2*Theta2';
    a3 = sigmoid(z3);
    h = a3;
    
fprintf('\n Size h \n'); size(h) 

a = -sum(sum(y_matrix.*log(h)));
fprintf('\n Size a \n'); size(a)
a

b = -sum(sum((1-y_matrix).*log(1-h)));
fprintf('\n Size b \n'); size(b)
b

regTheta1 = Theta1(:,2:end).^2;
regTheta2 = Theta2(:,2:end).^2;
regularization = lambda*(sum(sum(regTheta1)) + sum(sum(regTheta2)))/2;

J = ((a + b) + regularization)/m;

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

a1 = Xt;
fprintf('\n BackProp a1 \n'); size(a1)
z2 = a1*Theta1';
fprintf('\n BackProp a1 \n'); size(z2)
a2 = sigmoid(z2);
fprintf('\n BackProp a2 \n'); size(a2)
a2 = [ones(size(a2,1),1) a2];
fprintf('\n BackProp a2 \n'); size(a2)
z3 = a2*Theta2';
fprintf('\n BackProp z3 \n'); size(z3)
a3 = sigmoid(z3);
fprintf('\n BackProp a3 \n'); size(a3)

h = a3;
d3 = h - y_matrix;
fprintf('\n BackProp d3 \n'); size(d3)
d2 = (d3*Theta2(:,2:end)).*sigmoidGradient(z2);
fprintf('\n BackProp d2 \n'); size(d2)

Delta1 = d2'*a1;
fprintf('\n BackProp Delta1 \n'); size(Delta1)
Delta2 = d3'*a2;
fprintf('\n BackProp Delta2 \n'); size(Delta2)

Theta1(:,1) = 0;
Theta2(:,1) = 0;

Theta1_grad = (Delta1)/m;
Theta2_grad = (Delta2)/m;

Theta1_grad = Theta1_grad + (lambda/m)*Theta1;
Theta2_grad = Theta2_grad + (lambda/m)*Theta2;

% -------------------------------------------------------------
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
