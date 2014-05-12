function [ nn ] = nn_adjust( nn )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

load('../../GitHub/Deep-Learning/Coletti-DeepLearnToolbox/data/mnist_uint8.mat');

train_x = double(train_x) / 255;
train_y = double(train_y);

% normalize
[train_x, ~, ~] = zscore(train_x);

dropoutRate = nn.dropoutFraction;
inputDropoutRate = nn.inputCorruptFraction;
dropoutMean = 0.5;

nn.testing = 1;
nn = nnff(nn, train_x, train_y);
nn.testing = 0;
% nn.adjusting = 0;
nn.avgA = [];
nn.adj = [];

% for all units except visible units
for i = 1:(nn.n - 1)
    
    nn.avgA(i) = mean(mean(nn.a{i}));
    if (i == 1)
        nn.adj(1) = ( ( (1 - dropoutRate) * nn.avgA(i) ) + inputDropoutRate ...
            * dropoutMean ) / nn.avgA(i);
    else
        nn.adj(i) = ( ( (1 - dropoutRate) * nn.avgA(i) ) + dropoutRate * ... 
            dropoutMean ) / nn.avgA(i);
    end
%     nn.Wadj{i} = nn.W{i} .* norm_factor;
    
end


end

