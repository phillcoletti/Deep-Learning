function nn = nnsetup(architecture, initialization)
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
    nn.inputCorruptFraction             = 0;                %  Used for Denoising AutoEncoders
    nn.dropoutFraction                  = zeros(1,nn.n - 2);%  Dropout level for hidden layers        
    nn.testing                          = 0;                %  Internal variable. nntest sets this to one.
    nn.output                           = 'softmax';           %  output unit 'sigm' (=logistic), 'softmax' and 'linear'
    nn.initialization                   = initialization; % random or pretraining
    nn.numPredict                       = 1;              % Number of runs for testing

    %% Dropout training setup (See "Improving neural networks by preventing co-adaptation of feature detectors" by Hinton et al.)
    nn.dropoutTraining                  = 0;                %  Dropout training flag
    nn.dropoutInput                     = 0;                %  Dropout level for input layer
    nn.initialMomentum                  = 0.5;              %  First value of momentum
    nn.finalMomentum                    = 0.99;             %  Last value of momentum
    nn.endMomentumTime                  = 500;              %  Number of epochs during which momentum increases
    nn.maxSquaredLength                 = 15;               %  Constraint on incoming weight vector
    
    %% DropConnect training setup (See Regularization of Neural Networks using DropConnect)
    nn.connectTraining                  = 0;                %  DropConnect training flag
    nn.epochSchedule                    = [800,600,400,100,50,20,20];    %  size of each epoch segment (for different learning rates)
%    nn.epochSchedule                    = [1,1,1,1,1,1,1];    %  toy example for testing
    nn.learningRateMultiplier           = [1,.5,.1,.05,.01,.005, .001];    % learning rate multiplier
    nn.sigma                            = 0.25;             % Gaussian distortion
    
    %% Weight initialization
    
    if strcmp('random', initialization)
        for i = 2 : nn.n   
            % weights and weight momentum
            nn.W{i - 1} = (rand(nn.size(i), nn.size(i - 1)+1) - 0.5) * 2 * 4 * sqrt(6 / (nn.size(i) + nn.size(i - 1)));
            nn.vW{i - 1} = zeros(size(nn.W{i - 1}));
%             size(nn.W{i-1}) % DEBUG

            % average activations (for use with sparsity)
            nn.p{i}     = zeros(1, nn.size(i));   
        end
    elseif strcmp('pretraining', initialization)
        maxepoch = 50;
        numhid = nn.size(2);
        makebatches;
        [numcases numdims numbatches]=size(batchdata);
%         size(batchdata, 3)
%         max(max(max(batchdata)))
%         min(min(min(batchdata)))
%         mean(mean(mean(batchdata)))
%         batchdata = normalize01(batchdata);
        
        
        [vishid, hidbiases, visbiases, batchposhidprobs] = rbmf(maxepoch, numhid, batchdata, 1);
        nn.W{1} = [hidbiases; vishid]'; % check that this is the right format
%         size(nn.W{1}) % DEBUG
%         save('vishid2.mat', 'vishid');
        batchdata = batchposhidprobs;
        
        for i = 3:(nn.n - 1)
            numhid = nn.size(i);
            [vishid, hidbiases, visbiases, batchposhidprobs] = rbmf(maxepoch, numhid, batchdata, 1);
            nn.W{i - 1} = [hidbiases; vishid]';
%             size(nn.W{i - 1}) % DEBUG
            batchdata = batchposhidprobs;
        end
        
        % weights to softmax layer are not pretrained; they're randomly
        % initialized
        nn.W{nn.n - 1} = (rand(nn.size(nn.n), nn.size(nn.n - 1)+1) - 0.5) * 2 * 4 * sqrt(6 / (nn.size(nn.n) + nn.size(nn.n - 1)));
        size(nn.W{nn.n - 1})
        
        for i = 2 : nn.n   
            % weights and weight momentum
%             nn.W{i - 1} = (rand(nn.size(i), nn.size(i - 1)+1) - 0.5) * 2 * 4 * sqrt(6 / (nn.size(i) + nn.size(i - 1)));
            nn.vW{i - 1} = zeros(size(nn.W{i - 1}));

            % average activations (for use with sparsity)
            nn.p{i}     = zeros(1, nn.size(i));   
        end
        
        rng('default');
        rng shuffle;
    end
end
