function [ensembleMax, methods, ensembleSize] = get_ensembleMax()
addpath('../data/')

close all
%retrieve data
[ train_x, train_y, test_x, test_y ] = loadMNIST(  );
%get ensemble predictions
[ensembleErrors{1}, noises, corruptionRates] = get_EnsembleErrorProd('../results/',test_y);
ensembleErrors{2} = get_EnsembleErrorMean('../results/',test_y);
ensembleErrors{3} = get_EnsembleErrorMin('../results/',test_y);
ensembleErrors{4} = get_EnsembleErrorMax('../results/',test_y);
%ensembleErrorMedian = get_EnsembleErrorMedian('../results/',test_y);
methods = {'Product','Mean', 'Min', 'Max'};

for k = 1 : length(ensembleErrors)
    ensembleError = ensembleErrors{k};
    for i = 1 : length(noises)
        noise = noises{i};
        for j = 1 : length(corruptionRates)
            if ~isempty(ensembleError{i,j})
                error = ensembleError{i,j};
                ensembleMax(i,j,k) = error(end);
                ensembleSize(i,j) = length(ensembleError{i,j});
            end
        end
    end
end
        
        