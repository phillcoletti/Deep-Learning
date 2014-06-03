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
ensembleErrors{5} = ensembleError;
methods = {'Product','Mean', 'Min', 'Max', 'Median'};

for i = 1 : length(noises)
    noise = noises{i};
    for j = 1 : length(corruptionRates)
        corruptionRate = corruptionRates(j);
        figure;
        hold all;
        title(['Ensemble Polling Methods with ', noise, ' Noise, ' num2str(corruptionRate), ' Corruption Rate'])
        xlabel('Ensemble Size')
        ylabel('Error')
        legendTest = {};
        for k = 1 : length(ensembleErrors)
            ensembleError = ensembleErrors{k};
            if ~isempty(ensembleError{i,j})
                legendText{k} = [methods{k}, ' Polling'];
                error = ensembleError{i,j};
                n = length(error);
                plot(1:n,error, 'o-', 'LineWidth', 2);
            end
        end
        hold off;
        legend(legendText);
    end
end
        
        