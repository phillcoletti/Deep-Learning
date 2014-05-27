function [ A ] = get_network_prediction( nn )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

A = zeros(nn.a{end});
nn.testing = 0;
for i = 1:nn.numPredict
    nn = nnff(nn, train_x, train_y);
    A = A + nn.a{end};
end

A = A ./ nn.numPredict;

end

