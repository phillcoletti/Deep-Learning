function [error, labels] = ensemblePredict(dirpath, y)  
    files = dir(dirpath);
    for i = 1 : length(files)
        file = files(i);
        if ~strcmp(file.name, '.') && ~strcmp(file.name, '..')
            filename = [dirpath, file.name];
            predictions(:,:,i) = dlmread(filename, ' ');
        end
    end

    % Predict
    [~, expected] = max(y,[],2);    %How to get class for MNIST
    prob_y = mean(predictions, 3);
    [~, labels] = max(prob_y,[],2);
    bad = find(labels ~= expected);    
    error = numel(bad) / length(expected);
end