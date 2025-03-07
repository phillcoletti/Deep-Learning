function [nn, L, loss]  = nntrain(nn, train_x, train_y, test_x, test_y, modelnum, opts, val_x, val_y)
%NNTRAIN trains a neural net
% [nn, L] = nnff(nn, x, y, opts) trains the neural network nn with input x and
% output y for opts.numepochs epochs, with minibatches of size
% opts.batchsize. Returns a neural network nn with updated activations,
% errors, weights and biases, (nn.a, nn.e, nn.W, nn.b) and L, the sum
% squared error for each training minibatch.

assert(isfloat(train_x), 'train_x must be a float');
assert(nargin == 7 || nargin == 9,'number ofinput arguments must be 7 or 9')

loss.train.e               = [];
loss.train.e_frac          = [];
loss.test.e                = [];
loss.test.e_frac           = [];
loss.val.e                 = [];
loss.val.e_frac            = [];
opts.validation = 0;
if nargin == 9
    opts.validation = 1;
end

fhandle = [];
if isfield(opts,'plot') && opts.plot == 1
    fhandle = figure();
end

if nn.connectTraining
    trainingType = 'connect';
else
    trainingType = 'hinton';
end

% Write errors to file in case people decide to be mean and shut me down
fid = fopen(strcat('../results3/', nn.noise , '_', nn.activation_function, '_dropout=',num2str(nn.dropoutFraction),'_inputCorrupt=',num2str(nn.inputCorruptFraction), '_initialization=', nn.initialization, '_#', num2str(modelnum), '.txt'),'wt');

batchsize = opts.batchsize;
numepochs = opts.numepochs;

m = size(train_x, 1);
numbatches = m / batchsize;
assert(rem(numbatches, 1) == 0, 'numbatches must be a integer');

corruptions = {'drop', 'salt_pepper', 'random'};
L = zeros(numepochs*numbatches,1);
n = 1;
for i = 1 : numepochs
    
    if (nn.randCorruption)
        nn.noise = corruptions{ randi([1, 3]) };
    end
    
    % Dropout training: update momentum and learning rate
    if (nn.dropoutTraining)
        if i < nn.endMomentumTime
            nn.momentum = (1-i/(nn.endMomentumTime))*nn.initialMomentum + i/(nn.endMomentumTime)*nn.finalMomentum;
        else
            nn.momentum = nn.finalMomentum;
        end

        nn.learningRate = nn.learningRate * nn.scaling_learningRate;
    end
    
    tic;
    
    kk = randperm(m);
    for l = 1 : numbatches
        batch_x = train_x(kk((l - 1) * batchsize + 1 : l * batchsize), :);
        
        %Add noise to input (for use in denoising autoencoder)
        if(nn.inputCorruptFraction ~= 0)
            switch nn.noise 
                case 'drop'
                    batch_x = batch_x.*(rand(size(batch_x))>nn.inputCorruptFraction);
                case 'salt_pepper'
                    rand_units = rand(size(batch_x));
                    white_units = rand_units < (nn.inputCorruptFraction / 2);
                    black_units = (rand_units > (nn.inputCorruptFraction / 2)) & (rand_units < nn.inputCorruptFraction);
                    batch_x(white_units) = 0;
                    batch_x(black_units) = 1;
                case 'random'
                    rand_units = rand(size(batch_x)) < nn.inputCorruptFraction;
                    rand_mask = rand(size(batch_x));
                    batch_x(rand_units) = rand_mask(rand_units);
                case 'gaussian'
                    rand_units = rand(size(batch_x)) < nn.inputCorruptFraction;
                    batch_x = batch_x + (normrnd(0,nn.sigma,size(batch_x)) .* rand_units);
            end
        end
        
        batch_y = train_y(kk((l - 1) * batchsize + 1 : l * batchsize), :);
        
        nn = nnff(nn, batch_x, batch_y);
        nn = nnbp(nn);
        nn = nnapplygrads(nn);
        
        L(n) = nn.L;
        
        n = n + 1;
    end
    
    t = toc;

    if ~mod(i, 1)
        if opts.validation == 1
            loss = nnevalAVG(nn, loss, train_x, train_y, test_x, test_y, val_x, val_y, nn.numPredict);
            str_perf = sprintf('; Train mse = %f; Test MSE = %f; Val MSE = %f; Train Err = %f; Test Err = %f; Val Err = %f', loss.train.e(end), loss.test.e(end), loss.val.e(end), loss.train.e_frac(end), loss.test.e_frac(end), loss.val.e_frac(end));
        else
            loss = nnevalAVG(nn, loss, train_x, train_y, test_x, test_y, nn.numPredict);
            str_perf = sprintf('; Train MSE = %f; Test MSE = %f; Train Err = %f; Test Err = %f', loss.train.e(end), loss.test.e(end), loss.train.e_frac(end), loss.test.e_frac(end));
        end
        if ishandle(fhandle)
            nnupdatefigures(nn, fhandle, loss, opts, i);
        end

        results_str = ['epoch ' num2str(i) '/' num2str(numepochs) '. Took ' num2str(t) 's' '. Mini-batch MSE ' num2str(mean(L((n-numbatches):(n-1)))) str_perf '\n'];
        disp(results_str);

        % Write training error to file
        fprintf(fid, results_str);
    end


    %save intermediate neural network
    if ~mod(i, 200)
        if (nn.randCorruption)
            nn.noise = 'randCorrupt';
        end
        varname = strcat('../results3/', trainingType, '_', nn.noise , '_', nn.activation_function, '_dropout=',num2str(nn.dropoutFraction),'_inputCorrupt=',num2str(nn.inputCorruptFraction), '_initialization=', nn.initialization, '_#', num2str(modelnum), '_epochs=', num2str(i), '.mat');
        save(varname,'nn');
    end

end
fclose(fid);
end

