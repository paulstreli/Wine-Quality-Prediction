function [mse] = linearreg(XTRAIN, ytrain, XTEST, ytest, modulspec)
%% this function was generated to allow cross-validation for linear regression

% each training set is normalised after the split and the cross-validation set is normalised with the training parameters
[XTRAIN_norm,mu,sigma] = zscore(XTRAIN);
ytrain_mean = mean(ytrain);
ytrain_cent = ytrain-ytrain_mean;

% find linear regression model and mean squared error
linearregmdl = fitlm(XTRAIN_norm, ytrain_cent, modulspec);
XTEST_norm = (XTEST-mu)./sigma;
mse = immse(ytest, predict(linearregmdl, XTEST_norm)+ytrain_mean);
end


