function [ C, counts ] = get_nn_confusionMat( nn_preds, test_y )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

C = zeros( size(test_y, 2) );

[~, I] = max(nn_preds, [], 2);
% P = zeros( size(I, 1), size(test_y, 2) );
% 
% for i=1:size(I, 1)
%     P(i,I(i)) = 1;
% end

% size(I,1)
for i=1:size(I, 1)
    
%     find(test_y(i,:))
%     I(i)
    
    C( find(test_y(i,:)), I(i) ) = C( find(test_y(i,:)), I(i) ) + 1;
    
end

% C

counts = sum(C, 2);
for i=1:size(C, 1)
    
    C(i,:) = C(i,:) ./ sum( C(i,:) );
    
end

end

