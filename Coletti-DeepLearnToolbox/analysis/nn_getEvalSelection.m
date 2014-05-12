function [ mean_err, std_err, ensemble_err ] = nn_getEvalSelection( nns, train_x, train_y, test_x, test_y )
%nn_getEvalIndividual
%   returns the mean and std of the error rate for all models of a given
%   type (from the same directory)

addpath('../NN/');

numModels = length(nns);
[numEx, numClasses] = size(test_y);
error = zeros(1,numModels);
prob_y = zeros(numEx, numClasses, numModels);

%loop through all models and collect data
for i = 1 : numModels
    nn = nns{i};
    [err, ~] = nntest(nn, test_x, test_y);
    model_error(i) = err;
    nn.testing = 1;
    nn = nnff(nn, test_x, zeros(size(test_y)));
    prob_y(:,:, i) = nn.a{end};
end

%individual models
mean_err = mean(model_error);
std_err = std(model_error);

%ensemble
[~, ensemble_y] = max(sum(prob_y,3),[],2);
[~, expected] = max(test_y,[],2);
bad = find(ensemble_y ~= expected);    
ensemble_err = numel(bad) / length(test_y);