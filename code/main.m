%% SET UP

% close windows and clear workspace
close all
clear
clc

% set random seed to a fixed value for reproducibility
rng(9999999);

%% IMPORT OF WINE DATA AND PREPARATION FOR FURTHER USE

% generate training and test data for red wines, white wines and combined datasets and store them in tables
[data_comb_test_t, data_comb_train_t, data_white_test_t, data_white_train_t, data_red_test_t, data_red_train_t] = winedataimport();

% convert tables to matrices for use in functions requiring matrix-input
comb_test = table2array(data_comb_test_t); % matrix containing all test data 
comb_train = table2array(data_comb_train_t); % matrix containing all training data

% measure number of samples in test/training set
num_comb_test = size(comb_test,1); % number of test samples
num_comb_train = size(comb_train,1); % number of training samples

% split input features from outputs and store them in separate matrices
comb_train_in = data_comb_train_t{:,1:end-1}; % training input features
comb_train_out = data_comb_train_t.quality; % training outputs

comb_test_in = comb_test(:,1:end-1); % test input features
comb_test_out = comb_test(:,end); % test outputs

%% NORMALISATION: CREATE MATRICES WITH STANDARDISED FEATURES AND CENTERED OUTPUTS
% mu_train: vector containing the mean of each training input feature
% sigma_train: standard deviations of input features
[comb_train_in_norm,mu_train,sigma_train] = zscore(comb_train_in); 

% find the mean of the training outputs
mean_out = mean(comb_train_out)
comb_train_out_cent = comb_train_out - mean_out;

% scale/normalise the test set with the features calculated above 
comb_test_in_norm = (comb_test_in- mu_train)./sigma_train;
comb_test_out_cent = comb_test_out - mean_out;

%% CONSTANT BASELINE PREDICTOR
% create vectors storing the mean of the training outputs to ease calculation of mean-squared-error
mean_train_vector(1:num_comb_train) = mean_out;
mean_test_vector(1:num_comb_test) = mean_out;

% calculate the training and test mean-squared-error using the training output mean as constant base predictor
train_mse_base = immse(comb_train_out, mean_train_vector') % training error
test_mse_base = immse(comb_test_out, mean_test_vector') % test error

% calculate a 10-fold-cross-validation-error for later comparison
cvmean = @(XTRAIN,ytrain,XTEST)(mean(ytrain)*ones(size(XTEST,1),1));

cv_mse_base = crossval('mse',comb_train_in, comb_train_out,'predfun',cvmean)

%% LINEAR REGRESSION without feature transformation
% calculate the 10-fold-crossvalidation-error of the linear regression model without regularisation
linearreghandler = @(XTRAIN, ytrain, XTEST, ytest)(linearreg(XTRAIN, ytrain, XTEST, ytest, 'linear'));
cv_mse_linreg = mean(crossval(linearreghandler,comb_train_in, comb_train_out),1)

% train final linear regression model on whole dataset
linreg_model = fitlm(comb_train_in_norm, comb_train_out_cent)
test_mse_linreg = immse(comb_test_out_cent, predict(linreg_model, comb_test_in_norm))

%% RIDGE REGRESSION without feature transformation

% use 10-fold-crossvalidation to find the optimal lambda for ridge regression
% the non-normalised inputs and outputs are passed as they are normalised after splitting for cross-validation
lambda_ridge = 1000000 * (0.5).^ (0:50);
ridgereghandler = @(XTRAIN, ytrain, XTEST, ytest)(ridgereg(XTRAIN, ytrain, XTEST, ytest,lambda_ridge));
cv_mse_ridge = mean(crossval(ridgereghandler,comb_train_in, comb_train_out),1);

% extract the optimal lambda
[cv_mse_ridge_min, cv_mse_ridge_min_index] = min(cv_mse_ridge);
lambda_ridge_opt = lambda_ridge(cv_mse_ridge_min_index)
cv_mse_ridge_min

% plot the cross-validation error with varying lamda values
figure;
semilogx((lambda_ridge), cv_mse_ridge)
xlabel("lambda value");
ylabel("Cross-validation error");
title("Cross-validation error of ridge regression using different lambda values");

% train over the whole training data set with the optimal lambda and calculate ridge regression test error
ridge_coeff = ridge(comb_train_out_cent, comb_train_in_norm, lambda_ridge_opt);
test_mse_ridge = immse(comb_test_out_cent,(comb_test_in_norm*ridge_coeff))

%% LASSO without feature transformation
% use 10-fold-crossvalidation to find the optimal lambda for lasso
[lasso_coeff, lasso_model_inf] =lasso(comb_train_in_norm, comb_train_out_cent, 'CV', 10,'Standardize', 1, 'NumLambda', 50,'LambdaRatio', 1e-7);


% plot the cross-validation error with varying lamda values
lassoPlot(lasso_coeff, lasso_model_inf, 'PlotType', 'CV')

% extract the optimal lambda
lambda_lasso_opt = lasso_model_inf.LambdaMinMSE
cv_mse_lasso_min = min(lasso_model_inf.MSE)

% train over the whole training data set with the optimal lambda and calculate lasso test error
[lasso_coeff_final, lasso_model_final_inf] =lasso(comb_train_in_norm, comb_train_out_cent,'Standardize', 1, 'Lambda', [lambda_lasso_opt]);
test_mse_lasso = immse(comb_test_out_cent, (comb_test_in_norm*lasso_coeff_final))

%% ELASTIC NET without feature transformation
% use 10-fold-crossvalidation to find the optimal lambda and alpha for elastic net
mse_elastic_min = 99999999999;
for alpha_elastic = 0.00000001 * (1.4454).^ (0:50) % test different alpha values
    [lasso_elastic_coeff, lasso_elastic_inf] =lasso(comb_train_in_norm, comb_train_out_cent, 'CV', 10,'Standardize', 1, 'NumLambda', 50,'LambdaRatio', 1e-7, 'Alpha', alpha_elastic);
    if lasso_elastic_inf.MSE(lasso_elastic_inf.IndexMinMSE) < mse_elastic_min % if cross-validation error improves, store corresponding settings
        mse_elastic_min = lasso_elastic_inf.MSE(lasso_elastic_inf.IndexMinMSE);
        lasso_elastic_coeff_opt = lasso_elastic_coeff;
        lasso_elastic_inf_opt = lasso_elastic_inf;
        alpha_elastic_opt = alpha_elastic;
    end
end

% display optimal coefficients and cross-validation-error
alpha_elastic_opt
lambda_elastic_opt = lasso_elastic_inf_opt.LambdaMinMSE
cv_mse_elastic_min = mse_elastic_min

% train over the whole training data set with the optimal lambda and alphaa and calculate elastic net test error
[elastic_coeff_final, elastic_model_final_inf] =lasso(comb_train_in_norm, comb_train_out_cent,'Standardize', 1, 'Lambda', [lambda_elastic_opt], 'Alpha', alpha_elastic_opt);
test_mse_elastic = immse(comb_test_out_cent, (comb_test_in_norm*elastic_coeff_final))

%% CHECKING FOR NEW FEATURES

lin_reg_interaction = fitlm(comb_train_in_norm, comb_train_out_cent, 'interactions')
lin_reg_quadratic = fitlm(comb_train_in_norm, comb_train_out_cent, 'quadratic')

%% INTERACTION FEATURE GENERATION
% generate a training matrix with original and products of individual features ("interaction") 
comb_train_interact_in = interactiongen(comb_train_in);

% normalise new matrix
[comb_train_interact_in_norm, mu_interact, sigma_interact] = zscore(comb_train_interact_in);

% generate a test interaction matrix and normalise it with training parameters
comb_test_interact_in = interactiongen(comb_test_in);
comb_test_interact_in_norm = (comb_test_interact_in- mu_interact)./sigma_interact;

%% LINEAR REGRESSION with interaction features

%calculate the cross-validation error for
cv_mse_linreg_int = mean(crossval(linearreghandler,comb_train_interact_in, comb_train_out),1)

% train final linear regression model on whole dataset
linreg_int_model = fitlm(comb_train_interact_in_norm, comb_train_out_cent)
test_mse_linreg = immse(comb_test_out_cent, predict(linreg_int_model, comb_test_interact_in_norm))

%% RIDGE REGRESSION with interaction features
% use 10-fold-crossvalidation to find the optimal lambda for ridge regression
lambda_ridge_int = 1000000 * (0.5).^ (0:50);
ridgereghandler_int = @(XTRAIN, ytrain, XTEST, ytest)(ridgereg(XTRAIN, ytrain, XTEST, ytest,lambda_ridge_int));
cv_mse_ridge_int = mean(crossval(ridgereghandler_int,comb_train_interact_in, comb_train_out),1);

% extract the optimal lambda
[cv_mse_ridge_min_int, cv_mse_ridge_min_index_int] = min(cv_mse_ridge_int);
lambda_ridge_int_opt = lambda_ridge_int(cv_mse_ridge_min_index_int)
cv_mse_ridge_min_int

% plot the cross-validation error with varying lamda values
figure;
semilogx((lambda_ridge_int), cv_mse_ridge_int)
xlabel("lambda value");
ylabel("Cross-validation error");
title("Cross-validation error of ridge regression with interaction features using different lambda values");

% train over the whole training data set with the optimal lambda and calculate ridge regression test error
ridge_coeff_int = ridge(comb_train_out_cent, comb_train_interact_in_norm, lambda_ridge_int_opt);
test_mse_ridge_int = immse(comb_test_out_cent,(comb_test_interact_in_norm*ridge_coeff_int))

%% LASSO with interaction features

[lasso_coeff_int, lasso_model_int_inf] =lasso(comb_train_interact_in_norm, comb_train_out_cent, 'CV', 10,'Standardize', 1, 'NumLambda', 50,'LambdaRatio', 1e-7);

% plot the cross-validation error with varying lamda values
lassoPlot(lasso_coeff_int, lasso_model_int_inf, 'PlotType', 'CV')

% extract the optimal lambda
lambda_lasso_int_opt = lasso_model_int_inf.LambdaMinMSE
cv_mse_lasso_int_min = min(lasso_model_int_inf.MSE)

% train over the whole training data set with the optimal lambda and calculate lasso test error
[lasso_int_coeff_final, lasso_int_model_final_inf] =lasso(comb_train_interact_in_norm, comb_train_out_cent,'Standardize', 1, 'Lambda', [lambda_lasso_int_opt]);
test_mse_lasso_int = immse(comb_test_out_cent, (comb_test_interact_in_norm*lasso_int_coeff_final))

%% ELASTIC NET with interaction features
% use 10-fold-crossvalidation to find the optimal lambda and alpha for elastic net
mse_elastic_int_min = 99999999999;
for alpha_elastic_int = [1e-03 1e-1 0.3 0.5 0.7 0.9 1-(1e-03)] % test different alpha values
    [lasso_elastic_int_coeff, lasso_elastic_int_inf] =lasso(comb_train_interact_in_norm, comb_train_out_cent, 'CV', 10,'Standardize', 1, 'NumLambda', 20,'LambdaRatio', 1e-7, 'Alpha', alpha_elastic_int);
    if lasso_elastic_int_inf.MSE(lasso_elastic_int_inf.IndexMinMSE) < mse_elastic_int_min % if cross-validation error improves, store corresponding settings
        mse_elastic_int_min = lasso_elastic_int_inf.MSE(lasso_elastic_int_inf.IndexMinMSE);
        lasso_elastic_int_coeff_opt = lasso_elastic_int_coeff;
        lasso_elastic_int_inf_opt = lasso_elastic_int_inf;
        alpha_elastic_int_opt = alpha_elastic_int;
    end
    alpha_elastic_int
end

% display optimal coefficients and cross-validation-error
alpha_elastic_int_opt
lambda_elastic_int_opt = lasso_elastic_int_inf_opt.LambdaMinMSE
cv_mse_elastic_int_min = mse_elastic_int_min

% train over the whole training data set with the optimal lambda and alphaa and calculate elastic net test error
[elastic_int_coeff_final, elastic_int_model_final_inf] =lasso(comb_train_interact_in_norm, comb_train_out_cent,'Standardize', 1, 'Lambda', [lambda_elastic_int_opt], 'Alpha', alpha_elastic_int_opt);
test_mse_elastic_int = immse(comb_test_out_cent, (comb_test_interact_in_norm*elastic_int_coeff_final))

%% SVM

% use cross-validation to compare the performance of SVM with a gaussian and a linear Kernel
% can pass non-normalised inputs as fitrsvm function does standardisation for us
svm_mdl_gaussian = fitrsvm(comb_train_in, comb_train_out, 'KFold', 10, 'Standardize', true, 'KernelFunction', 'gaussian');
cv_mse_svm_gauss = kfoldLoss(svm_mdl_gaussian)
svm_mdl_linear = fitrsvm(comb_train_in, comb_train_out, 'KFold', 10, 'Standardize', true, 'KernelFunction', 'linear');
cv_mse_svm_linear = kfoldLoss(svm_mdl_linear)

% gaussian had better cross-validation error -> optimise hyperparameters, optimisation was carried out via function below
% function commented out as parameters were extracted after optimisation
%svm_mdl_gaussian_opt = fitrsvm(comb_train_in, comb_train_out, 'Standardize', true, 'KernelFunction', 'gaussian', 'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus'));

% find cross-validation error of optimised svm regression with gaussian-kernel 
cv_svm_mdl_gaussian_opt = fitrsvm(comb_train_in, comb_train_out, 'Standardize', true, 'KernelFunction','gaussian', 'Kfold', 10, 'Epsilon', 0.0012431, 'KernelScale', 2.8992, 'BoxConstraint', 1.6608);
cv_mse_gaussian_opt = kfoldLoss(cv_svm_mdl_gaussian_opt)

% find the corresponding test errors and the final hypothesis by training over the whole dataset without optimised parameters
svm_mdl_gaussian_final = fitrsvm(comb_train_in, comb_train_out, 'Standardize', true, 'KernelFunction', 'gaussian');
test_mse_svm_gauss = immse(comb_test_out,predict(svm_mdl_gaussian_final, comb_test_in))

% find the corresponding test errors and the final hypothesis by training over the whole dataset with optimised parameters
svm_mdl_gaussian_final_opt = fitrsvm(comb_train_in, comb_train_out, 'Standardize', true, 'KernelFunction', 'gaussian', 'Epsilon', 0.0012431, 'KernelScale', 2.8992, 'BoxConstraint', 1.6608);
test_mse_svm_gauss_opt = immse(comb_test_out,predict(svm_mdl_gaussian_final_opt, comb_test_in))
%% NEURAL NETWORK
% The QualityRegressionNeuralNetwork function was trained, cross-validated and created using the built-in matlab NeuralNetwork-GUI (nnstart)
% Determine the test error using neural network
neural_network_out = QualityRegressionNeuralNetwork(comb_test_in_norm);
test_mse_neural = immse(comb_test_out_cent, neural_network_out)

%% BAGGER TREES
% Use Cross-validation to find correct number of trees (normalisation is done after cross-validation split in treebaggermdl function)
mdl_tree_handler_100 = @(XTRAIN, ytrain, XTEST, ytest)(treebaggermdl(XTRAIN, ytrain, XTEST, ytest, 100));
cv_mse_tree_100 = mean(crossval(mdl_tree_handler_100, comb_train_in, comb_train_out),1)
mdl_tree_handler_10 = @(XTRAIN, ytrain, XTEST, ytest)(treebaggermdl(XTRAIN, ytrain, XTEST, ytest, 10));
cv_mse_tree_10 = mean(crossval(mdl_tree_handler_10, comb_train_in, comb_train_out),1)

% Determine final model and test error for 100 bagged trees (best result)
mdl_tree_final = TreeBagger(100, comb_train_in_norm, comb_train_out_cent, 'Method','regression');
test_mse_tree =immse(predict(mdl_tree_final, comb_test_in_norm), comb_test_out_cent)
