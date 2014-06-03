addpath('../data/')

close all
%retrieve data
[ train_x, train_y, test_x, test_y ] = loadMNIST(  );
%get ensemble predictions
%[ensembleErrorProd, noises, corruptionRates] = get_EnsembleErrorProd('../results/',test_y);
%ensembleErrorMean = get_EnsembleErrorMean('../results/',test_y);
%ensembleErrorMin = get_EnsembleErrorMin('../results/',test_y);
%ensembleErrorMax = get_EnsembleErrorMax('../results/',test_y);
%ensembleErrorMedian = get_EnsembleErrorMedian('../results/',test_y);

j = 2; % 0.5 corrupt rate
for i = 1 : length(noises)
    noise = noises{i};
    figure;
    hold all;
    title(['Ensemble Polling Methods with ', noise, ' Noise'])
    xlabel('Ensemble Size')
    ylabel('Error')
    n = length(ensembleErrorMean{i,j});
    plot(1:n, ensembleErrorMean{i,j}, 'o-', 'LineWidth', 2)
    plot(1:n, ensembleErrorMedian{i,j}, 'o-', 'LineWidth', 2)
    plot(1:n, ensembleErrorMin{i,j}, 'o-', 'LineWidth', 2)
    plot(1:n, ensembleErrorMax{i,j}, 'o-', 'LineWidth', 2)
    plot(1:n, ensembleErrorProd{i,j}, 'o-', 'LineWidth', 2)
    hold off;
    legend('Mean', 'Median', 'Min', 'Max', 'Product');
end
        
        