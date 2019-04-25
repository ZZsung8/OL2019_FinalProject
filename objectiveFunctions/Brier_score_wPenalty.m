
function F = Brier_score_wPenalty(w,L,X,Y)
% Inputs:
% X is an m x n features matrix
% Y is an m x 1 label vector
% w is an n x 1 weights vector
%
% Output:
% F is a scalar (F1_score)

m = size(X,1); % Number of beats
n = size(X,2); % Number of features
Xw = X*w;            % Matrix-vector multiplication
expXw = exp(-Xw);    % Exponential of X*w
Y = double(Y);
F = 1/m*sum((Y-1./(1+expXw)).^2) + L*sum(w.*w);
end
