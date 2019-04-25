function pred_prob = sig_predict(w,X)
pred_prob = 1./(1+exp(-X*w)); % Prediction probability for w and X
end