function [nn, L, loss, er, bad] = test_dropout_rates(inputdropout,rates, numepochs)
load mnist_uint8;

train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);

% normalize
[train_x, mu, sigma] = zscore(train_x);
test_x = normalize(test_x, mu, sigma);

%% Neural net with dropout
rand('state',0);
nn = nnsetup([784 800 800 10]);

nn.dropoutTraining = 1;
nn.learningRate = 10;
nn.scaling_learningRate = 0.998;
opts.batchsize = 100;

nn.dropoutInput = inputdropout;
nn.dropoutFraction = rates;
opts.numepochs =  numepochs;                %  Number of full sweeps through data
                   
[nn, L, loss] = nntrain(nn, train_x, train_y, opts);

[er, bad] = nntest(nn, test_x, test_y);
save(strcat('dropoutrate',num2str(inputdropout),regexprep(num2str(rates),'[^\w'']',''),'.mat'),'nn','L','er','bad', 'loss');
% strcat(noise,'_epochs=', num2str(numepochs), '_dropout=',num2str(rates),'_inputdrop=',num2str(inputdropout),'.mat')