addpath('../data/')

%retrieve data
[ train_x, train_y, test_x, test_y ] = loadMNIST(  );
%get ensemble predictions
[ensembleError, noises, corruptionRates] = get_EnsembleError('../wacky_results/',test_y);


for i = 1 : length(noises)
    noise = noises{i};
    figure;
    hold all;
    title(['Ensemble Error with ', noise, ' Noise'])
    xlabel('Ensemble Size')
    ylabel('Error')
    for j = 1 : length(corruptionRates)
        corruptionRate = corruptionRates(j);
        legendText{i} = ['Corruption Rate = ', num2str(corruptionRate)];
        error = ensembleError{i,j};
        n = len(error);
        plot(1:n,error);
    end
    hold off;
    legend(legendText{i});
end
        
        