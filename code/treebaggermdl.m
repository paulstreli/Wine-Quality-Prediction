function [mse] = treebaggermdl(XTRAIN, ytrain, XTEST, ytest, numberoftrees)
%% this function was generated to allow cross-validation for baggertrees

% each training set is normalised after the split and the cross-validation set is normalised with the training parameters
[XTRAIN_norm,mu,sigma] = zscore(XTRAIN);
ytrain_mean = mean(ytrain);
ytrain_cent = ytrain-ytrain_mean;

% find bagged tree model and mean squared error
treebaggermdl = TreeBagger(numberoftrees, XTRAIN_norm, ytrain_cent, 'Method','regression'); 
XTEST_norm = (XTEST-mu)./sigma;
mse = immse(ytest, predict(treebaggermdl, XTEST_norm)+ytrain_mean);
end