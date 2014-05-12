function [ NNX ] = get_nns_mapping_matrix( dirName )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

fnames = getAllFiles(dirName);
% fnames{1}
% size(fnames,1)
NNX = zeros(size(fnames, 1), 1000);

for i = 1:size(fnames, 1)
    load(fnames{i});
    NNV = get_nn_mapping_vector(nn);
    NNX(i,:) = NNV;
end

end

