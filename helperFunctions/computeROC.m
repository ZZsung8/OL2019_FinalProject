function [sens, spec] = computeROC(alpha,X,Y,w)
% Computes ROC curve for given X,Y, and weights w
predY_prob = sig_predict(w,X); % Predicted label probabilities
Y = double(Y); % In case Y is not double precision
sens = zeros(length(alpha),1);
spec = zeros(length(alpha),1);
for a = 1:length(alpha)
   A = alpha(a); % Use this as the decision rule
   sens(a) = sum(Y.*(predY_prob>=A))/sum(Y); % Compute the sensitivity
   spec(a) = sum( (1-Y).*((predY_prob<A)))/sum(1-Y); % Compute the specificity
end
end
