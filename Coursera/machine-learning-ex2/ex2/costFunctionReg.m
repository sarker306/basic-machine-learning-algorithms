function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta



function r = h(X, i)
r = sigmoid(X(i, :) * theta);
end

for i=1:m,
    J += y(i) * log(h(X, i)) + (1 - y(i)) * log(1 - h(X, i));
end;

J = -J / m;

% You should not regularize theta(1)
n = length(theta);
for j=2:n,
  J += lambda * theta(j) * theta(j) / (2*m);
end

n = length(theta);
for j=1:n,
    grad(j) = 0;
    for i=1:m,
        grad(j) += (h(X,i) - y(i)) * X(i,j);
    end;
    grad(j) /= m;
    if (j > 1)
      grad(j) += lambda * theta(j) / m;
    endif
end;

% =============================================================

end