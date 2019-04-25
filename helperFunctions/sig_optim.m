function f = sig_optim(w,X,Y,n)
% y is a vector
% X is a matrix
% w is a vector
% n specifies which constraint number this is
Y = double(Y); % Y is a logical vector
Xw = X*w; % Matrix vector multiplication between X and weights
YXw = Y.*Xw;
expXw = Y./(1+exp(-YXw));
f = (-1/length(Y))*X(1,n)*expXw(1); % Initialize with the first entry
for i = 2:size(X,2) % Loop through features
    f = f + (-1/length(Y))*X(i,n)*expXw(i);
end

end