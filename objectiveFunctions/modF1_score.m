% Written by Eric Sung
% modF1_score.m
% Modified F1-score objective function
% Because the F1-score is notoriously difficult
% to optimize, we use a modified F1-score that instead
% of comparing predictions to true labels, compares
% prediction PROBABILITIES to true labels. Thus we
% have a continuous space to optimize over

function F = modF1_score(w,L,X,Y)
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
TP = Y.*(1./(1+expXw));
FN = Y.*(expXw./(1+expXw));
TN = (1-Y).*(expXw./(1+expXw));
FP = (1-Y).*(1./(1+expXw));

P = mean(TP(~isnan(TP)))/(mean(TP(~isnan(TP)))...
        +mean(FP(~isnan(FP)))); % Precision
R = mean(TP(~isnan(TP)))/(mean(TP(~isnan(TP)))...
        +mean(FN(~isnan(FN)))); % Recall
F = 1- 2*P*R/(P+R);    % The modified F1 score except we want to minimize it
end
