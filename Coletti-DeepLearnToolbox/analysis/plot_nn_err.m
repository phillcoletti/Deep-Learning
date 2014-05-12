function [ ] = plot_nn_err( train_err, test_err, corruption_type )
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here

plot(1:size(train_err, 1), train_err, 'Color', 'g');
hold on;
plot(1:size(test_err, 1), test_err, 'Color', 'b');
xlabel('epochs');
ylabel('error');
title_str = sprintf('Training and test error %s', corruption_type);
title(title_str);
legend('Training error', 'Test error');

end

