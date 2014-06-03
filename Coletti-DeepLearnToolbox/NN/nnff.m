function nn = nnff(nn, x, y)
%NNFF performs a feedforward pass
% nn = nnff(nn, x, y) returns an neural network structure with updated
% layer activations, error and loss (nn.a, nn.e and nn.L)
    
    % get weight adjustment for salt_peppr and random cases
    % called here instead of in loop
%     nn = nn_adjust(nn);
    
    n = nn.n;
    m = size(x, 1);
    
    x = [ones(m,1) x];
    nn.a{1} = x;
    
    % HANH CODE
    %dropout for input layer
%     if ~(nn.testing)
%         nn.dropOutMask{1} = (rand(size(nn.a{1}))>nn.dropoutInput);
%         nn.a{1} = x.*nn.dropOutMask{1};
%     end
    
    
    %feedforward pass
    for i = 2 : n-1
        switch nn.activation_function 
            case 'sigm'
                % Calculate the unit's outputs (including the bias term)
                nn.a{i} = sigm(nn.a{i - 1} * nn.W{i - 1}');
            case 'tanh_opt'
                nn.a{i} = tanh_opt(double(nn.a{i - 1}) * nn.W{i - 1}');
            case 'relu'
                nn.a{i} = relu(nn.a{i - 1} * nn.W{i - 1}');    
        end
        
        %dropout
        if(nn.dropoutFraction > 0)
            if(nn.testing)
                switch nn.noise
                    case 'drop'
                        nn.a{i} = nn.a{i} * (1 - nn.dropoutFraction);
                    case {'salt_pepper', 'random'}
                        nn.a{i} = nn.a{i} * nn.adj(i);
                end
%                 if strcmp(nn.noise,'drop')
%                     nn.a{i} = nn.a{i}.*(1 - nn.dropoutFraction);
%                 end
            else
%                if ~(nn.adjusting)
                    rand_units = rand(size(nn.a{i}));
                    nn.dropOutMask{i} = rand_units >= nn.dropoutFraction;
                    switch nn.noise 
                        case 'drop'
                            nn.a{i} = nn.a{i} .* nn.dropOutMask{i};
                        case 'salt_pepper'
                            white_units = rand_units < (nn.dropoutFraction / 2);
                            black_units = (rand_units >= (nn.dropoutFraction / 2)) & (rand_units < nn.dropoutFraction);
                            nn.a{i}(white_units) = 0;
                            nn.a{i}(black_units) = 1;
                        case 'random'
                            rand_mask = rand(size(nn.a{i}));
                            nn.a{i}(~nn.dropOutMask{i}) = rand_mask(~nn.dropOutMask{i});
                        case 'gaussian'
                            nn.a{i} = nn.a{i} + (normrnd(0,nn.sigma,size(nn.a{i})) .* ~nn.dropOutMask{i});
                    end
%                end
            end
        end
        
        %calculate running exponential activations for use with sparsity
        if(nn.nonSparsityPenalty>0)
            nn.p{i} = 0.99 * nn.p{i} + 0.01 * mean(nn.a{i}, 1);
        end
        
        %Add the bias term
        nn.a{i} = [ones(m,1) nn.a{i}];
    end
    switch nn.output 
        case 'sigm'
            nn.a{n} = sigm(nn.a{n - 1} * nn.W{n - 1}');
        case 'linear'
            nn.a{n} = nn.a{n - 1} * nn.W{n - 1}';
        case 'softmax'
            nn.a{n} = nn.a{n - 1} * nn.W{n - 1}';
            nn.a{n} = exp(bsxfun(@minus, nn.a{n}, max(nn.a{n},[],2)));
            nn.a{n} = bsxfun(@rdivide, nn.a{n}, sum(nn.a{n}, 2)); 
    end

    %error and loss
    nn.e = y - nn.a{n};
%     nn.e(1)
%     m
    
    switch nn.output
        case {'sigm', 'linear'}
            nn.L = 1/2 * sum(sum(nn.e .^ 2)) / m; 
        case 'softmax'
%             nn.L = -sum(sum(y .* log(nn.a{n}))) / m;
            nn.L = -sum(sum(y .* log(nn.a{n} + eps))) / m;
    end
end
