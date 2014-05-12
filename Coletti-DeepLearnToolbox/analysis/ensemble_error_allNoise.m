% ensemble_error_allNoise.m
% Calculates model and ensemble error for one of each type of model, all
% types of noise

clear nns;

addpath('../results/');
addpath('../data/');

[ train_x, train_y, test_x, test_y ] = loadMNIST();

%load models
%load('../results/drop/nns/tanh/05/connect_drop_tanh_opt_dropout=0.5_inputCorrupt=0.2_initialization=random_#1_epochs=1990_FINAL.mat');
%nns{1} = nn;
load('../results/salt_pepper/nns/tanh/05/connect_salt_pepper_tanh_opt_dropout=0.5_inputCorrupt=0.2_initialization=random_#1_epochs=1990_FINAL.mat');
nns{1} = nn;
load('../results/random/nns/tanh/05/connect_random_tanh_opt_dropout=0.5_inputCorrupt=0.2_initialization=random_#2_epochs=1990_FINAL.mat');
nns{2} = nn;

[ mean_err, std_err, ensemble_err ] = nn_getEvalSelection( nns, train_x, train_y, test_x, test_y )