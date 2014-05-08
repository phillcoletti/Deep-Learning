function nn = nnapplygrads(nn)
%NNAPPLYGRADS updates weights and biases with calculated gradients
% nn = nnapplygrads(nn) returns an neural network structure with updated
% weights and biases
    
    for i = 1 : (nn.n - 1)
        if(nn.weightPenaltyL2>0)
            dW = nn.dW{i} + nn.weightPenaltyL2 * [zeros(size(nn.W{i},1),1) nn.W{i}(:,2:end)];
        else
            dW = nn.dW{i};
        end
        
        dW = nn.learningRate * dW;
        
        % Dropout momentum update
        if (nn.dropoutTraining)
            nn.vW{i} = nn.momentum*nn.vW{i} + (1-nn.momentum)*dW; % See Hinton et al
            dW = nn.vW{i};
        else
            if(nn.momentum>0)
                nn.vW{i} = nn.momentum*nn.vW{i} + dW;
                dW = nn.vW{i};
            end
        end
        
        nn.W{i} = nn.W{i} - dW;
        
        % Dropout weight constraint
        if (nn.dropoutTraining) && (nn.maxSquaredLength > 0)
            incomingW = sum(nn.W{i}.^2,2);
            scaling = (incomingW <= nn.maxSquaredLength) + (incomingW > nn.maxSquaredLength)*nn.maxSquaredLength./incomingW;
            nn.W{i} = nn.W{i}.*repmat(scaling,1,size(nn.W{i},2));
        end
    end
end
