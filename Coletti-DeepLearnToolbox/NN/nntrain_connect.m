function [nn, L, loss]  = nntrain_connect(nn, train_x, train_y, test_x, test_y, modelnum, opts, val_x, val_y)
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
fid = fopen(strcat('../results/', trainingType, '_', nn.noise , '_', nn.activation_function, '_dropout=',num2str(nn.dropoutFraction),'_inputCorrupt=',num2str(nn.inputCorruptFraction), '_initialization=', nn.initialization, '_#', num2str(modelnum), '.txt'),'wt');

batchsize = opts.batchsize;

m = size(train_x, 1);
numbatches = m / batchsize;
assert(rem(numbatches, 1) == 0, 'numbatches must be a integer');

maxEpochs = sum(nn.epochSchedule);
L = zeros(maxEpochs*numbatches,1);
n = 1;

numepochs = 0;
currEpoch = 1;
for k = 1 : length(nn.epochSchedule)
    numepochs = numepochs + nn.epochSchedule(k);
    nn.learningRate = nn.learningRate * nn.learningRateMultiplier(k);       %scale the learning rate down with schedule
        
    for i = currEpoch : numepochs

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

        if opts.validation == 1
            loss = nneval(nn, loss, train_x, train_y, test_x, test_y, val_x, val_y);
            str_perf = sprintf('; Train mse = %f; Test MSE = %f; Val MSE = %f; Train Err = %f; Test Err = %f; Val Err = %f', loss.train.e(end), loss.test.e(end), loss.val.e(end), loss.train.e_frac(end), loss.test.e_frac(end), loss.val.e_frac(end));
        else
            loss = nneval(nn, loss, train_x, train_y, test_x, test_y);
            str_perf = sprintf('; Train MSE = %f; Test MSE = %f; Train Err = %f; Test Err = %f', loss.train.e(end), loss.test.e(end), loss.train.e_frac(end), loss.test.e_frac(end));
        end
        if ishandle(fhandle)
            nnupdatefigures(nn, fhandle, loss, opts, i);
        end
        
        results_str = ['epoch ' num2str(i) '/' num2str(maxEpochs) '. Took ' num2str(t) 's' '. Mini-batch MSE ' num2str(mean(L((n-numbatches):(n-1)))) str_perf '\n'];
        disp(results_str);

        %save intermediate neural network
        if ~mod(i, 1) 
            varname = strcat('../results/', trainingType, '_', nn.noise , '_', nn.activation_function, '_dropout=',num2str(nn.dropoutFraction),'_inputCorrupt=',num2str(nn.inputCorruptFraction), '_initialization=', nn.initialization, '_#', num2str(modelnum), '_epochs=', num2str(numepochs), '.mat');
            save(varname,'nn');
        end

        % Write training error to file
        fprintf(fid, results_str);

    end
    currEpoch = i+1;
    nn.learningRate = nn.learningRate / nn.learningRateMultiplier(k);   %rescale the learning rate to original for next multiplier
end
fclose(fid);
end

