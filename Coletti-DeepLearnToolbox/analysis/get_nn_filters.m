function [ I ] = get_nn_filters( nn )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

% get first layer weights
W = nn.W{1};

% get rid of bias term
W = W(:,2:end);

% normalize [0, 1]
W = (W - min(min(W))) ./ (max(max(W)) - min(min(W)));

% perhaps pick subset of filters with the greatest diff between max and min
% to find interesting ones

% most papers extract 64 (in an 8 x 8 grid)

[~, IDX] = sort(range(W, 2), 'descend');
% size(IDX)
% IDX(1:80)

I = zeros(231);

for i = 1:64
%     I = vec2mat(W(i,:), 28);
    F = reshape(W(IDX(i),:), [28 28]);
    row = ceil(i/8);
    col = mod(i, 8);
    I_row = (row - 1) * 29 + 1;
    I_col = col * 29 + 1;
    I(I_row:(I_row + 27), I_col:(I_col + 27)) = F;
end

end

