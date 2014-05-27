function [ ] = dir_nnPredictN( dirName, N )
%NN_GETPREDICT Summary of this function goes here
%   Detailed explanation goes here

addpath('../data/');
[ train_x, train_y, test_x, test_y ] = loadMNIST(  );

fnames = getAllFiles(dirName);
% fnames{1}
% size(fnames,1)

for i = 1:size(fnames, 1)
    fname = fnames{i};
    disp(['Predicting: ', fname])
    predname = strrep(fname, 'FINAL', '');
    predname = strrep(predname, '.mat', 'PREDICTIONS.mat');
    load(fnames{i});
    [prob_y, labels, error] = nn_findAvgPredictN(nn, N, test_x, test_y);
    save(predname, 'prob_y','labels','error');
end


end

