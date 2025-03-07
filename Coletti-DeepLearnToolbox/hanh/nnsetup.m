function nn = nnsetup(architecture)
%NNSETUP creates a Feedforward Backpropagate Neural Network
% nn = nnsetup(architecture) returns an neural network structure with n=numel(architecture)
% layers, architecture being a n x 1 vector of layer sizes e.g. [784 100 10]

    nn.size   = architecture;
    nn.n      = numel(nn.size);
    
    nn.activation_function              = 'tanh_opt';       %  Activation functions of hidden layers: 'sigm' (sigmoid) or 'tanh_opt' (optimal tanh).
    nn.learningRate                     = 2;                %  learning rate Note: typically needs to be lower when using 'sigm' activation function and non-normalized inputs.
    nn.momentum                         = 0.5;              %  Momentum
    nn.scaling_learningRate             = 1;                %  Scaling factor for the learning rate (each epoch)
    nn.weightPenaltyL2                  = 0;                %  L2 regularization
    nn.nonSparsityPenalty               = 0;                %  Non sparsity penalty
    nn.sparsityTarget                   = 0.05;             %  Sparsity target
    nn.inputZeroMaskedFraction          = 0;                %  Used for Denoising AutoEncoders
    nn.testing                          = 0;                %  Internal variable. nntest sets this to one.
    nn.output                           = 'sigm';           %  output unit 'sigm' (=logistic), 'softmax' and 'linear'

    %% Dropout training setup (See "Improving neural networks by preventing co-adaptation of feature detectors" by Hinton et al.)
    nn.dropoutTraining                  = 0;                %  Dropout training flag
    nn.dropoutInput                     = 0;                %  Dropout level for input layer
    nn.dropoutFraction                  = zeros(1,nn.n - 2);%  Dropout level for hidden layers        
    nn.initialMomentum                  = 0.5;              %  First value of momentum
    nn.finalMomentum                    = 0.99;             %  Last value of momentum
    nn.endMomentumTime                  = 500;              %  Number of epochs during which momentum increases
    nn.maxSquaredLength                 = 15;               %  Constraint on incoming weight vector
    
    %% Weight initialization
    for i = 2 : nn.n   
        % weights and weight momentum
        nn.W{i - 1} = (rand(nn.size(i), nn.size(i - 1)+1) - 0.5) * 2 * 4 * sqrt(6 / (nn.size(i) + nn.size(i - 1)));
        nn.vW{i - 1} = zeros(size(nn.W{i - 1}));
        
        % average activations (for use with sparsity)
        nn.p{i}     = zeros(1, nn.size(i));   
    end
end
