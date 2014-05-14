function [error, labels, names] = nn_findAvgPredict2(N, x, y)
% Plots the prediction of the network using nnpredictAVG as a function of
% the number of "models" averaged. This is how we derive accurate
% predictions for noise other than dropout

clear nns;

addpath('../results/');
addpath('../NN/');

[~, expected] = max(y,[],2);

%load models
% %drop=0.5
% load('../results/drop/nns/tanh/05/connect_drop_tanh_opt_dropout=0.5_inputCorrupt=0.2_initialization=random_#1_epochs=1990_FINAL.mat');
% nns{1} = nn;
% names{1} = 'drop, rate = 0.5';
% load('../results/salt_pepper/nns/tanh/05/connect_salt_pepper_tanh_opt_dropout=0.5_inputCorrupt=0.2_initialization=random_#1_epochs=1990_FINAL.mat');
% nns{2} = nn;
% names{2} = 'salt_pepper, rate=0.5';
% load('../results/random/nns/tanh/05/connect_random_tanh_opt_dropout=0.5_inputCorrupt=0.2_initialization=random_#2_epochs=1990_FINAL.mat');
% nns{3} = nn;
% names{3} = 'random, rate = 0.5';
%drop=0.25
load('../results/drop/nns/tanh/025/connect_drop_tanh_opt_dropout=0.25_inputCorrupt=0.2_initialization=random_#1_epochs=1990_FINAL.mat');
nns{1} = nn;
names{1} = 'drop, rate = 0.25';
load('../results/salt_pepper/nns/tanh/025/connect_salt_pepper_tanh_opt_dropout=0.25_inputCorrupt=0.2_initialization=random_#1_epochs=1990_FINAL.mat');
nns{2} = nn;
names{2} = 'salt_pepper, rate=0.25';
load('../results/random/nns/tanh/025/connect_random_tanh_opt_dropout=0.25_inputCorrupt=0.2_initialization=random_#5_epochs=1990_FINAL.mat');
nns{3} = nn;
names{3} = 'random, rate = 0.25';
% %drop=0.75
% load('../results/drop/nns/tanh/075/connect_drop_tanh_opt_dropout=0.75_inputCorrupt=0.2_initialization=random_#1_epochs=1990_FINAL.mat');
% nns{7} = nn;
% names{7} = 'drop, rate = 0.75';
% load('../results/salt_pepper/nns/tanh/075/connect_salt_pepper_tanh_opt_dropout=0.75_inputCorrupt=0.2_initialization=random_#1_epochs=1990_FINAL.mat');
% nns{8} = nn;
% names{8} = 'salt_pepper, rate=0.75';
% load('../results/random/nns/tanh/075/connect_random_tanh_opt_dropout=0.75_inputCorrupt=0.2_initialization=random_#5_epochs=1990_FINAL.mat');
% nns{9} = nn;
% names{9} = 'random, rate = 0.75';

numModels = length(nns);
varname = [];

for k = 1 : numModels
    disp(names{k})
    varname = [varname, '__', names{k}];
    nns{k}.y_predictions = zeros(size(x,1), nn.size(end), numModels);
    for i = 1 : N
        nns{k} = nnff(nns{k}, x, zeros(size(x,1), nns{k}.size(end)));
        nns{k}.y_predictions(:,:,i) = nns{k}.a{end};
        prob_y = mean(nns{k}.y_predictions, 3);
        [~, labels(:,i,k)] = max(prob_y,[],2);
        bad = find(labels(:,i,k) ~= expected);    
        er = numel(bad) / size(x, 1);
        error(i,k) = er;
    end
    interim_varname = [varname, '__numIter', num2str(N), '.mat'];
    save(interim_varname, 'names','error','labels');
end

varname = [varname, '__numIter', num2str(N), '.mat'];

save(varname, 'names','error','labels');