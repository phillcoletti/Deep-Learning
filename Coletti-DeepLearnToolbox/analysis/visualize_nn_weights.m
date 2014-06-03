function [ G ] = visualize_nn_weights( nn, layer, train_x, train_y )
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here

addAllPaths;

if (layer == 1)
    I = get_nn_filters(nn);
else
    
    if ~( isfield(nn, 'numPredict') )
        nn.numPredict = 1;
    end
    
    A = zeros( size(train_x, 1), size(nn.a{layer + 1}, 2) );
    for i = 1:nn.numPredict
        nn = nnff(nn, train_x, train_y);
        A = A + nn.a{layer + 1};
    end
    A = A ./ nn.numPredict;
    
    I = zeros( size(nn.a{layer + 1}, 2), size(train_x, 2) );
    
%     for i = 1:size(nn.a{layer + 1}, 2)
    for i = 1:64
        
%         output = strcat('unit=', i);
%         disp(output);
        
        a_mat = repmat(A(:,i), [1, size(train_x, 2)]);
        i_mat = train_x .* a_mat;
        I(i,:) = sum(i_mat, 1) ./ sum( a_mat(:,1) );
        
    end
    
%     a_mat
    
    G = zeros(231);
    for i = 1:64
        
        I_i = reshape(I(i,:), [28, 28])';
        row = ceil(i / 8);
        col = mod(i, 8);
        I_row = (row - 1) * 29 + 1;
        I_col = col * 29 + 1;
        G(I_row:(I_row + 27), I_col:(I_col + 27)) = I_i;
        
    end
    
%     for i = 1:size(train_x, 1)
%         I = I + A(i,1) .* train_x(i,:);
%     end
%     I = I ./ sum(A(:,1));
%     I = reshape(I, [28, 28])';
    
end

end

