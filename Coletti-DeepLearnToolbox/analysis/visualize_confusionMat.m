function [ I ] = visualize_confusionMat( C )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

addAllPaths;

I = log(round(10000 * C) + eps);
I(I < 0) = 0;
I = normalize01(I);
I = imresize(I, 5);

end

