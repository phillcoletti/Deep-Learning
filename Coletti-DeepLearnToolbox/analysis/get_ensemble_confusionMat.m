function [ CM, CSD ] = get_ensemble_confusionMat( dirName, test_y )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

fnames = getAllFiles(dirName);

classes = size(test_y, 2);
C_ENS = zeros( classes, classes, size(fnames, 1) );

for i = 1:size(fnames, 1)
    load(fnames{i});
    [C_i, ~] = get_nn_confusionMat(prob_y, test_y);
    C_ENS(:,:,i) = C_i;
end

CM = mean(C_ENS, 3);
CSD = std(C_ENS, 0, 3);

end

