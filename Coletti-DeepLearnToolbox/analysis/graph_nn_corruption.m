function [ ] = graph_nn_corruption( dirList )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

X = [];
corruptTypeInds = [];

for i = 1:size(dirList, 2)
    NNX = get_nns_mapping_matrix(dirList{i});
    X = [X; NNX];
    corruptTypeInds = [corruptTypeInds, size(NNX, 1)];
end

corruptTypes = ['none', 'drop', 'salt_pepper', 'gaussian', 'random'];
colors = {'m', 'b', 'r', 'g', 'c'};

% note that tSNE has issues when there are more dimensions than there are
% examples, so typically PCA is used beforehand. Here, since we only have
% 20 examples (5 models per 4 corruption types), we project to 20
% dimensions first. Hinton projected to 30 in his introduction to the
% technique.
% mappedPCA = compute_mapping(X, 'PCA', 20);
% size(X)
% vec1 = X(1, :);
% vec2 = X(2, :);
% all(vec1 == vec2)
mappedPCA = compute_mapping(X, 'PCA', size(X, 1));
mappedA = compute_mapping(mappedPCA, 'tSNE', 2);

% scatter(mappedA(1:corruptTypeInds(1),1), mappedA(1:corruptTypeInds(1),2), ...
%     2, colors(1), 'fill');

scatter(mappedA(:,1), mappedA(:,2), 8, 'r', 'fill')

for i = 1:size(corruptTypes, 2)
    scatter(mappedA(1:corruptTypeInds(i),1), mappedA(1:corruptTypeInds(i),2), ...
        2, colors{i}, 'fill');
    hold on;
end

legend('none', 'drop', 'salt_pepper', 'gaussian', 'random');
title('tSNE visualization of network mapping and corruption');

end

