function labels = nnpredictAVG(nn, x, N)
    nn.testing = 0;
    y_predictions = zeros(size(x,1), nn.size(end), N);
    for i = 1 : N
        nn = nnff(nn, x, zeros(size(x,1), nn.size(end)));
        y_predictions(:,:,i) = nn.a{end};
    end
    prob_y = mean(y_predictions, 3);
    [dummy, i] = max(prob_y,[],2);
    labels = i;
end
