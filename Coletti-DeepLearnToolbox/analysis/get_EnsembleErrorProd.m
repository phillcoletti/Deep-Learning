function [ ensembleError, noises, corruptionRates ] = get_EnsembleErrorProd( dirName, y )
%PLOT_ENSEMBLESIZE Summary of this function goes here
%   Detailed explanation goes here

initializations = {'pretraining', 'random'};
activation = 'tanh_opt';
inputCorruptFraction = 0.2;
noises = {'drop', 'salt_pepper', 'random', 'mixed'};
corruptionRates = [0.4, 0.5, 0.6, 0.7];
dim = {'Corruption Rate', 'Noise', 'Ensemble Size'};

[~, expected] = max(y,[],2);

fnames = getAllFiles(dirName);

%Remove non-prediction files
N = length(fnames);
k = 1;
while k <= N
    disp(['searching ', fnames{k}])
    if isempty(strfind(fnames{k}, 'PREDICTIONS'))
        disp(['removing ', fnames{k}])
        fnames(k) = [];
        N = N-1;
    else
        k = k + 1;
    end
end

for i = 1 : length(noises)
    for j = 1 : length(corruptionRates)
        numModels = 0;
        ensembleError{i,j} = [];
        y_prob = zeros(size(y));        
        for a = 1 : length(initializations)
            N = length(fnames);
            k = 1;
            while k <= N
                initialization = initializations{a};
                noise = noises{i};
                corruptionRate = corruptionRates(j);
                fname = fnames{k};
                searchstr = [noise , '_',activation, '_dropout=',num2str(corruptionRate),'_inputCorrupt=',num2str(inputCorruptFraction), '_initialization=', initialization];
                if strfind(fname, searchstr)
                    disp(['loading: ', fname])
                    load(fname);
                    numModels = numModels + 1;
                    y_prob(:,:,numModels) = prob_y;
                    %remove fromt eh list of file names
                    fnames(k) = [];
                    N = N-1;
                else
                    k = k+1;
                end
            end
            for n = 1 : numModels
                comb = combnk(1:numModels, n);
                error = 0;
                for c = 1 : size(comb,1)
                    probComb = y_prob(:,:,comb(c,:));
                    %ensemble
                    [~, ensemble_y] = max(prod(probComb,3),[],2);
                    bad = find(ensemble_y ~= expected); 
                    error(c) = numel(bad) / length(y);
                end
                ensembleError{i,j}(n) = mean(error);
            end
        end
    end
end

end
