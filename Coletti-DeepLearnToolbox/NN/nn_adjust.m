function [ nn ] = nn_adjust( nn, x )
%nn_adjust
% calculates the adjustment ratio by which to multiply the activations

dropoutRate = nn.dropoutFraction;
inputDropoutRate = nn.inputCorruptFraction;
dropoutMean = 0.5;

nn.adjusting = 1;
nn = nnff(nn, x, 0);
nn.adjusting = 0;
nn.avgA = [];
nn.adj = [];

% for all units except visible units
for i = 1:(nn.n - 1)
    
    nn.avgA(i) = mean(mean(nn.a{i}));
    if (i == 1)
        nn.adj(1) = ( ( (1 - inputDropoutRate) * nn.avgA(i) ) + inputDropoutRate ...
            * dropoutMean ) ./ nn.avgA(i);
    else
        nn.adj(i) = ( ( (1 - dropoutRate) * nn.avgA(i) ) + dropoutRate * ... 
            dropoutMean ) ./ nn.avgA(i);
    end
%     nn.Wadj{i} = nn.W{i} .* norm_factor;
    
end
nn.adj(1) = 1;

end

