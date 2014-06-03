addpath('../data/')

close all
%retrieve data
[ train_x, train_y, test_x, test_y ] = loadMNIST(  );
%get ensemble predictions
[ensembleError, noises, corruptionRates] = get_EnsembleErrorMean('../results/',test_y);


for i = 1 : length(noises)
    noise = noises{i};
    figure;
    hold all;
    title(['Ensemble Error with ', noise, ' Noise'])
    xlabel('Ensemble Size')
    ylabel('Error')
    legendTest = {};
    for j = 1 : length(corruptionRates)
        corruptionRate = corruptionRates(j);
        if ~isempty(ensembleError{i,j})
            legendText{j} = ['Corruption Rate = ', num2str(corruptionRate)];
            error = ensembleError{i,j};
            n = length(error);
            plot(1:n,error, 'o-', 'LineWidth', 2);
        end
    end
    hold off;
    legend(legendText);
end

for j = 1 : length(corruptionRates)
    corruptionRate = corruptionRates(j);
    figure;
    hold all;
    title(['Ensemble Error with Corruption Rate = ', num2str(corruptionRate)])
    xlabel('Ensemble Size')
    ylabel('Error')
    legendTest = {};
    for i = 1 : length(noises)
        noise = noises{i};
        if ~isempty(ensembleError{i,j})
            legendText{i} = [noise, 'noise'];
            error = ensembleError{i,j};
            n = length(error);
            plot(1:n,error, 'o-', 'LineWidth', 2);
        end
    end
    hold off;
    legend(legendText);
end
        
        