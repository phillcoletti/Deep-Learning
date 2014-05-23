function [ NNV ] = get_nn_mapping_vector( nn )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

NNV = [];

for i = 0:9
    fname = sprintf('./testdata/T%d.mat', i);
    varname = sprintf('T%d', i);
    load(fname);
%     size(eval(varname))
    y = zeros(size(eval(varname), 1), 10);
    y(:,i+1) = 1;
    nn = nnff(nn, eval(varname), y);
%     size(nn.a{end})
    A = nn.a{end};
    A = A';
%     size(vertcat(A))
%     sum(A,2)
    NNV = [NNV; A(:)];
end


end

