function [nn, L, loss, er, bad] = test_connect(noise,inputCorrupt,dropoutRate,activation, modelnum)
addpath('../data');
addpath('../util/');
addpath('../NN/');
addpath('../results/');

load mnist_uint8;

train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);

% normalize
[train_x, mu, sigma] = zscore(train_x);
test_x = normalize(test_x, mu, sigma);

%% Neural net with dropout -- trained with parameters from DropConnect paper
rand('state', 0);
nn = nnsetup([784 800 800 10], 'random');

nn.connectTraining = 1;
nn.learningRate = .1;
nn.momentum = 0.9;
opts.batchsize = 100;

nn.noise = noise;
opts.sigma = .25;
nn.inputCorruptFraction = inputCorrupt;
nn.dropoutFraction = dropoutRate;
nn.activation_function = activation;

[nn, L, loss] = nntrain_connect(nn, train_x, train_y, test_x, test_y, modelnum, opts);

[er, bad] = nntest(nn, test_x, test_y);

%save final neural network
numepochs = sum(nn.epochSchedule);
varname = strcat('../results/', 'connect_', noise, '_', nn.activation_function, '_dropout=',num2str(dropoutRate),'_inputCorrupt=',num2str(inputCorrupt), '_#', num2str(modelnum), '_epochs=', num2str(numepochs), '_FINAL.mat');
save(varname,'nn','L','er','bad', 'loss');