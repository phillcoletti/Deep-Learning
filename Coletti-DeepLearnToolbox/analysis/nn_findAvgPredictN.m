function [prob_y, labels, error] = nn_findAvgPredictN(nn, N, x, y)
% Plots the prediction of the network using nnpredictAVG as a function of
% the number of "models" averaged. This is how we derive accurate
% predictions for noise other than dropout

addpath('../results/');
addpath('../NN/');

[~, expected] = max(y,[],2);

y_predictions = zeros(size(x,1), nn.size(end));
for i = 1 : N
    nn = nnff(nn, x, zeros(size(x,1), nn.size(end)));
    y_predictions(:,:,i) = nn.a{end};
end

prob_y = mean(y_predictions, 3);
[~, labels] = max(prob_y,[],2);
bad = find(labels ~= expected);    
error = numel(bad) / size(x, 1);