function [loss] = nneval(nn, loss, train_x, train_y, test_x, test_y, val_x, val_y)
%NNEVAL evaluates performance of neural network
% Returns a updated loss struct
assert(nargin == 6 || nargin == 8, 'Wrong number of arguments');

% training performance
nn                      = nnff(nn, train_x, train_y);
loss.train.e(end + 1)   = nn.L;
% nn.L
nn                      = nnff(nn, test_x, test_y);
loss.test.e(end+1) = nn.L;
% nn.L

% validation performance
if nargin == 8
    nn                    = nnff(nn, val_x, val_y);
    loss.val.e(end + 1)   = nn.L;
end

if strcmp(nn.output, 'softmax')
%calc misclassification rate if softmax
    [er_train, dummy]               = nntest(nn, train_x, train_y);
    loss.train.e_frac(end+1)        = er_train;
    
    [er_test, dummy]                = nntest(nn, test_x, test_y);
    loss.test.e_frac(end+1)         = er_test;
    
    if nargin == 8
        [er_val, dummy]             = nntest(nn, val_x, val_y);
        loss.val.e_frac(end+1)      = er_val;
    end
end

end
