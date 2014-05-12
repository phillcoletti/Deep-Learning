function [ X ] = extract_nn_err( fname )
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here

X = [];
fid = fopen(fname, 'r');
line = 1;

while (line ~= -1)
    
    line = fgets(fid);
%     line
    if (line ~= -1)
        LV = get_line_vals(line);
        X = [X; LV];
    end
    
end

end

function [LV] = get_line_vals(line)

% line
LV = zeros(1, 5);

epoch = regexp(line, 'epoch (\d+)', 'tokens');
epoch = epoch{1};
LV(1) = str2double(epoch);

train_mse = regexp(line, 'Train MSE = (\d+\.\d+)', 'tokens');
train_mse = train_mse{1};
LV(2) = str2double(train_mse);

test_mse = regexp(line, 'Test MSE = (\d+\.\d+)', 'tokens');
test_mse = test_mse{1};
LV(3) = str2double(test_mse);

train_err = regexp(line, 'Train Err = (\d+\.\d+)', 'tokens');
train_err = train_err{1};
LV(4) = str2double(train_err);

test_err = regexp(line, 'Test Err = (\d+\.\d+)', 'tokens');
test_err = test_err{1};
LV(5) = str2double(test_err);

end

