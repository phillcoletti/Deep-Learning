function [ mean_err, std_err ] = nn_getEvalIndividual( dirName, train_x, train_y, test_x, test_y )
%nn_getEvalIndividual
%   returns the mean and std of the error rate for all models of a given
%   type (from the same directory)

addpath('../NN/');

fnames = getAllFiles(dirName);
numModels = size(fnames, 1);
error = zeros(1,numModels);

%loop through all models and collect data
for i = 1 : numModels
    clear nn
    load(fnames{i});
    fnames{i}
    [model_err, ~] = nntest(nn, test_x, test_y);
    error(i) = model_err;
    model_err
end

mean_err = mean(error);
std_err = std(error);
