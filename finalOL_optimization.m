% Written by Eric Sung
% finalOL_optimization.m
% The main body of where the optimization algorithm is done
% We proceed in a few steps:
% (0) Load the data
% (1) Use random forests to extract top features and reduce the
%     dimensionality of our problem
% (~) Developed a bilevel optimization problem to find the optimal
%     hyperparameter for logistic regression (DID NOT WORK)
% (2) Instead of the bilevel optimization, try to do the old-fashioned
%     grid-search approach to figure out how sensitive is our particular
%     dataset to the hyperparameter
%           - To do this, we custom write all of our functions including
%           cross-validation, logistic regression using the Brier score as
%           an objective function, and plugging them all into MATLAB's
%           optimization toolbox
%  (3) Generates figures to be used in our presentation.

%% (0) Load in our data matrix
load('PVCdata.mat') % Loads in an X and Y variable
% X is an m x n features matrix extracted from the HCTSA package
% Y is an m x 1 vector with the labels for classification

% First split the data into a training and test set
p = 0.6; % This designates the percentage we would like our training set to be
num_train = round(size(X,1)*p); 
S = rng(); % Use the same random number generator
R = randperm(size(X,1));
train_idx = R(1:num_train);
test_idx = setdiff([1:size(X,1)],train_idx);
X_train = X(train_idx,:); Y_train = Y(train_idx);
X_test = X(test_idx,:); Y_test = Y(test_idx);

%% (1) To reduce the dimensionality of our problem, we'll run a random forest 
% and extract the top 50 features
already_ran = 0; % If we've already run it, don't do it again
num_tree = 200;
if already_ran == 0
    RF_Mdl = TreeBagger(num_tree,X_train,Y_train,'OOBPredictorImportance','on','Method',...
    'classification');
    save(['RF_Model_' num2str(num_tree) 'trees.mat'])
else
    load('RF_Model_200trees.mat') % This contains our original test and training sets too
end

% Compute the out-of-bag importance for each feature for each of the decision trees
% and use this as a surrogate of feature ranking.

impOOB = zeros(size(X_train,2),1);
for i = 1:num_tree
    impOOB = impOOB + predictorImportance(RF_Mdl.Trees{i})';
end

% Now use this to figure out which features are most important
[~,ranked_idx] = sort(impOOB,'descend');
num_feat = 50;                                  % Choose only the top 50
bestFeatures = ranked_idx(1:num_feat);          % Extract only the best of the features
save('bestFeatures_indices','bestFeatures')     % Save them

figure()
plot([1:size(X_train,2)]',impOOB,'r','linewidth',1.5)
hold on; plot(bestFeatures,impOOB(bestFeatures),'g*','linewidth',3);hold off
title('Out-of-Bag Importance Across all Trees','fontsize',24)
xlabel('Feature Index','fontsize',20)
ylabel('Out of Bag Importance','fontsize',20)
legend('Scores for all Features','Best Features','fontsize',16);

% Set up the reduced training matrix
bias_train = ones(size(X_train,1),1) ;
bias_test = ones(size(X_test,1),1);

X_train_red = X_train(:,bestFeatures);
X_train_red = [bias_train X_train_red];
Y_train_red = double(Y_train); % Just to keep naming conventions the same
X_test_red = X_test(:,bestFeatures);
X_test_red = [bias_test X_test_red];
Y_test_red = double(Y_test); % Reduce features in our test set too

%% (~) Set up our logistic regression optimization problem
% This area is code that DID NOT WORK
%
% A few possibilities for why it's not working:
% (1) The F1 score may not be a good objective function to optimize
%       - Although this may be true, even using the Brier score with the
%       constraints did not seem to work
% (2) The exponential function tends to cause numerical instability issues 
%     because of floating-point arithmetic. Thus, it's possible that when
%     the gradient is computed in these regions, it makes it very difficult
%     to interpret what direction to go
% (3) The MATLAB optimization algorithms may be ill-suited to tackle non-linear
%     optimization problems. However, even with different algorithms, I
%     arrive at similar non-convergent solutions


% First we'll set up our constraints
m = length(Y_train);
N = size(X_train_red,2);       % From the previous section
weight = optimvar('Weights',N);
lambda = optimvar('Regularizer',1,'LowerBound',0);
lowerLevel = optimconstr(N,1);

yXw_vect = cell(m,1); % Store our expressions in here
columns_vect = cell(N,1); % 
disp('Performing matrix vector multiplication on X and w...')

for n = 1:N
   columns_vect{n} = fcn2optimexpr(@sig_optim,weight,X_train_red,...
                        Y_train_red,n); 
end
disp('Finished matrix vector multiplication.')
for n = 1:N
    lowerLevel(n) = 2*weight(n)*lambda +columns_vect{n} == 0 ;
end

% Now set up the objective function F1 score
obj = fcn2optimexpr(@modF1_score,weight,lambda,X_train_red,Y_train_red);
prob = optimproblem('Objective',obj);
prob.Constraints.lowerLevel = lowerLevel;

% Solve our optimization problem
options = optimoptions('fmincon','Display','iter',...
            'FunctionTolerance',1e-6);
w0 = struct('Weights',zeros(N,1),'Regularizer',1);
[w1,~,~,~] = solve(prob,w0,'options',options);

% Instead of F1-score, try the Brier score objective function

obj2 = fcn2optimexpr(@Brier_score,weight,lambda,X_train_red,Y_train_red);
prob2 = optimproblem('Objective',obj2);
prob2.Constraints.lowerLevel = lowerLevel;
options = optimoptions('fmincon','Display','iter',...
            'FunctionTolerance',1e-6);
w0 = struct('Weights',zeros(N,1),'Regularizer',1);
[w2,~,~,~] = solve(prob2,w0,'options',options);
predProbs = sig_predict(w2.Weights,X_train_red);
accur = sum(Y_train_red == (predProbs>0.5))/length(Y_train_red);

%% (2) Grid search study sensitivity of hyper-parameter choice
% I could not get the bi-level optimization scheme to work nor could I get 
% the F1-score metric to converge. Thus, I decided to use the Brier score
% with a fixed penalty term instead. In essence, I do a custom grid-search
% where on each step, I run my optimization solver to find the optimal
% weights
%
% We perform a custom cross-validation scheme to make our results more
% robust to overfitting
% 
values = [0:.02:2]; % Try a grid search across the lambdas

cross_val = 10; % Number of cross_validation loops
partitions = hist(1:size(X_train_red,1),cross_val);
parts_start = [1 cumsum(partitions(1:cross_val-1))+1]; % Indices for partitions
parts_end = cumsum(partitions); % End indices
 
validate_accuracy = zeros(cross_val,length(values));
weights_cL = cell(cross_val,length(values)); % Store the weights for our Brier score

for c = 1:cross_val
    v = setdiff([1:cross_val],c); % Indices for the training set
    Xc_train = [];
    Yc_train = [];
    Xc_val = X_train_red(parts_start(c):parts_end(c),:); % Validation set
    Yc_val = Y_train_red(parts_start(c):parts_end(c));   % Validation labels
    for t = v
       Xc_train = [Xc_train;X_train_red(parts_start(t):parts_end(t),:)];
       Yc_train = [Yc_train;Y_train_red(parts_start(t):parts_end(t))];
    end
    for l_idx = 1:length(values)
        disp(['Lambda = ' num2str(values(l_idx)) ' Cross ' ...
            num2str(c) ' of ' num2str(cross_val)])
        L = values(l_idx); % Value of regularization parameter
        obj3 = fcn2optimexpr(@Brier_score_wPenalty,weight,lambda,Xc_train,Yc_train);
        prob3 = optimproblem('Objective',obj3);
        constraints3 = optimconstr(1);
        constraints3(1) = lambda == L;  % Fixed our regularization parameter
        prob3.Constraints.C = constraints3;
        options = optimoptions('fmincon',...
                    'FunctionTolerance',1e-6);
        w0 = struct('Weights',zeros(N,1),'Regularizer',L);
        [w3,fval1,exitflag1,output1] = solve(prob3,w0,'options',options);
        predProbs = sig_predict(w3.Weights,Xc_val);
        validate_accuracy(c,l_idx) = sum(Yc_val == (predProbs>0.5))/length(Yc_val);
        weights_cL{c,l_idx} = w3.Weights;
    end
end

%% (3) Making figures 

% First we'll make box-plots of our accuracies given our cross-validation
boxplot(validate_accuracy(:,1:5:end),values(1:5:end))
ylabel('Percentage Accuracy','fontsize',16)
xlabel('\lambda Weight Value','fontsize',16)
title('Brier Score Cross-Validation Accuracies','fontsize',18)

% Make some ROC curves for the different lambda's
% Threshold for decision rule:
values_orig =  [0:0.02:2]; % original values used
values = values_orig(1:10:end);
alpha = 0:.01:1; % use these as our thresholds for classifying something as a normal beat
figure();
hold on;
colors = ['crbgym'];
legend_labels = cell(length(values)+1,1);
for l_idx = 1:length(values)
    w_l = mean([weights_cL{1:cross_val,l_idx}],2); % Take the mean weight
    [sens,spec] = computeROC(alpha,X_train_red,Y_train_red,w_l);
    plot(1-spec,sens,colors(mod(l_idx,length(colors))+1),'linewidth',3);
    legend_labels{l_idx} = ['\lambda = ' num2str(values(l_idx))];
end
plot(0:.01:1,0:.01:1,'k--','linewidth',2)
legend_labels{length(values)+1} = 'Reference Line';
hold off
xlabel('1-Specificity','fontsize',20)
ylabel('Sensitivity','fontsize',20)
title('ROC Curves for Various \lambda s for Training','fontsize',24)
legend(legend_labels,'fontsize',16)

% Generate figures for our testing data
alpha = 0:.01:1; % use these as our thresholds for classifying something as a normal beat
figure();
hold on;
colors = ['crbgym'];
legend_labels = cell(length(values)+1,1);
for l_idx = 1:length(values)
    w_l = mean([weights_cL{1:cross_val,l_idx}],2); % Take the mean weight
    [sens,spec] = computeROC(alpha,X_test_red,Y_test_red,w_l);
    plot(1-spec,sens,colors(mod(l_idx,length(colors))+1),'linewidth',3);
    legend_labels{l_idx} = ['\lambda = ' num2str(values(l_idx))];
end
plot(0:.01:1,0:.01:1,'k--','linewidth',2)
legend_labels{length(values)+1} = 'Reference Line';
hold off
xlabel('1-Specificity','fontsize',20)
ylabel('Sensitivity','fontsize',20)
title('ROC Curves for Various \lambda s for Testing','fontsize',24)
legend(legend_labels,'fontsize',16)