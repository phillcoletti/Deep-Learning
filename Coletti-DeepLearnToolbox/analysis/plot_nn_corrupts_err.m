function [  ] = plot_nn_corrupts_err( TE_ERR )
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here

corruptTypes = ['none', 'drop', 'salt_pepper', 'gaussian', 'random'];
colors = {'m', 'b', 'r', 'g', 'c'};
% size(TE_ERR,1)

for i = 1:5
%     colors{i}
    plot(1:size(TE_ERR, 1), TE_ERR(:,i), 'Color', colors{i});
    hold on;  
end

title('Test error and corruption type');
xlabel('epochs');
ylabel('test error');
legend('none', 'drop', 'salt pepper', 'gaussian', 'random');

end

