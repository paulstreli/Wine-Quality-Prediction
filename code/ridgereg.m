function [mse] = ridgereg(XTRAIN, ytrain, XTEST, ytest,lambda_ridge)
%% function for using cross-validation on finding the correct lambda value for ridgeregression
% output is a vector with mean-squared errors - each corresponding to a specific lambda value

% normalise the training set after the split
[XTRAIN_norm,mu,sigma] = zscore(XTRAIN);
ytrain_mean = mean(ytrain);
ytrain_cent = ytrain-ytrain_mean;

% calculate the ridge regression coefficients for specific lambda values (lambda_ridge is a vector)
coeff = ridge(ytrain_cent, XTRAIN_norm, lambda_ridge);

% normalise the validation set with the parameters from the training set
XTEST_norm = (XTEST-mu)./sigma;

% calculate the mean-squared error on the validation set
mse = (ones(1,length(ytest))*((ytest-(XTEST_norm*coeff+ytrain_mean)).^2))./length(ytest);
end

