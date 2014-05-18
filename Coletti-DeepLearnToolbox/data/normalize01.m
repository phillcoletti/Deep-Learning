function [ Y ] = normalize01( X )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% size(X)

% 2D matrix
if ( (size(X,1) > 1) && (size(X,2) > 1) ) 
    if (size(X, 3) == 1)
        Y = (X - min(min(X))) ./ (max(max(X)) - min(min(X)));
    
    % 3D matrix
    else
        Y = (X - min(min(min(X)))) ./ (max(max(max(X))) ...
        - min(min(min(X))));
    end

% 1D matrix (vector)
else
    Y = (X - min(X)) ./ (max(X) - min(X));
end

end

