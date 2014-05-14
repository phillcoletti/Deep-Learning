function [ ] = plot_nn_err( error, rate )
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here

hold all;
for i = 1 : size(error,2)
    plot(1:100, error(1:100,i))
end
hold off;
xlabel('# Predictions Averaged');
ylabel('Error');
title_str = ['Test Error, Corruption Rate = ', num2str(rate)];
title(title_str);
legend('Dropout', 'Salt & Pepper', 'Random');

end

