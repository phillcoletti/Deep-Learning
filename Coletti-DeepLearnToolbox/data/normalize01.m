function [ Y ] = normalize01( X )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

Y = (X - min(min(X))) ./ (max(max(X)) - min(min(X)));

end

