function modelnum = getModelnum( filename )
%GETMODELNUM Summary of this function goes here
%   Detailed explanation goes here

i = 1;
    while exist(strcat(filename, num2str(i), '.txt'), 'file')
        i = i + 1;
    end
    modelnum = i;
end