function [nn, L, loss, er, bad] = test_hinton(noise,inputCorrupt,dropoutRate,activation,initialization, numepochs, modelnum)
addpath('../data');
addpath('../util/');
addpath('../NN/');
addpath('../NN/Autoencoder_Code');
addpath('../results/');

% load normalized MNIST data
[ train_x, train_y, test_x, test_y ] = loadMNIST();

%% Neural net with dropout
% rand('state',0);
rng shuffle;
nn = nnsetup([784 800 800 10], initialization);

nn.randCorruption = 0;
if strcmp(noise, 'randCorrupt')
   nn.randCorruption = 1; 
end

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
% opts.validation = 0;

[nn, L, loss] = nntrain(nn, train_x, train_y, test_x, test_y, modelnum, opts);

[er, bad] = nntest(nn, test_x, test_y);

%save final neural network

varname = strcat('../results3/', 'hinton_', noise, '_', nn.activation_function, '_dropout=',num2str(dropoutRate),'_inputCorrupt=',num2str(inputCorrupt), '_initialization=', initialization, '_#', num2str(modelnum), '_epochs=', num2str(numepochs), '_FINAL.mat');
save(varname,'nn','L','er','bad', 'loss');