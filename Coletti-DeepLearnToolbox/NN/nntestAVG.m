function [er, bad] = nntest(nn, x, y, N)
    labels = nnpredictAVG(nn, x, N);
    [dummy, expected] = max(y,[],2);
    bad = find(labels ~= expected);    
    er = numel(bad) / size(x, 1);
end
