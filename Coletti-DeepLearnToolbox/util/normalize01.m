function [ normX ] = normalize01( X )
%NORMALIZE01 Summary of this function goes here
%   Detailed explanation goes here

normX = (X - min(min(X))) ./ (max(max(X)) - min(min(X)));

end

