function [ train_x, train_y, test_x, test_y ] = loadMNIST(  )
%PREPAREMNIST loads and normalizes the mnist dataset
addpath('../util/');

load mnist_uint8;

train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);

% normalize
[train_x, mu, sigma] = zscore(train_x);
test_x = normalize(test_x, mu, sigma);

end

