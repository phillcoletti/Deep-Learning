function [ er, bad, cnn ] = test_example_CNN( )
addpath('../data');
addpath('../util/');
addpath('../NN/');
addpath('../NN/Autoencoder_Code');
addpath('../results/');
addpath('../CNN/');

% load mnist_uint8;
% 
% train_x = double(reshape(train_x',28,28,60000))/255;
% test_x = double(reshape(test_x',28,28,10000))/255;
% train_y = double(train_y');
% test_y = double(test_y');

[ train_x, train_y, test_x, test_y ] = loadMNIST();
train_x = reshape(train_x', 28, 28, 60000);
test_x = reshape(test_x', 28, 28, 10000);
train_y = train_y';
test_y = test_y';

%% ex1 Train a 6c-2s-12c-2s Convolutional neural network 
%will run 1 epoch in about 200 second and get around 11% error. 
%With 100 epochs you'll get around 1.2% error

rand('state',0)

cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %subsampling layer
};


opts.alpha = 1;
opts.batchsize = 50;
opts.numepochs = 200;

cnn = cnnsetup(cnn, train_x, train_y);
cnn = cnntrain(cnn, train_x, train_y, opts);

[er, bad] = cnntest(cnn, test_x, test_y);
cnn.er = er;
cnn.bad = bad;
save('MNIST_CNN.mat', 'cnn');

%plot mean squared error
% figure; plot(cnn.rL);
% assert(er<0.12, 'Too big error');
