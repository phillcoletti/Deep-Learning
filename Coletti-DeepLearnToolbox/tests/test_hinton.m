function [nn, L, loss, er, bad] = test_hinton(noise,inputCorrupt,dropoutRate,activation,initialization, numepochs, modelnum)
addpath('../data');
addpath('../util/');
addpath('../NN/');
addpath('../results/');

% load normalized MNIST data
loadMNIST();

%% Neural net with dropout
% rand('state',0);
rng shuffle;
nn = nnsetup([784 800 800 10], initialization);

% nn.initialization = initialization;
nn.dropoutTraining = 1;
nn.connectTraining = 0;
nn.learningRate = 10;
nn.scaling_learningRate = 0.998;
opts.batchsize = 100;

nn.noise = noise;
opts.sigma = .25;
nn.inputCorruptFraction = inputCorrupt;
nn.dropoutFraction = dropoutRate;
nn.activation_function = activation;
% 3000 in Hinton paper
opts.numepochs =  numepochs;                %  Number of full sweeps through data

[nn, L, loss] = nntrain(nn, train_x, train_y, modelnum, opts, 0, 0);

[er, bad] = nntest(nn, test_x, test_y);

%save final neural network

varname = strcat('../results/', 'hinton_', noise, '_', nn.activation_function, '_dropout=',num2str(dropoutRate),'_inputCorrupt=',num2str(inputCorrupt), '_initialization=', initialization, '_#', num2str(modelnum), '_epochs=', num2str(numepochs), '_FINAL.mat');
save(varname,'nn','L','er','bad', 'loss');