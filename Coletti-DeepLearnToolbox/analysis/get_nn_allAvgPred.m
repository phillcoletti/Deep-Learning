addpath('../data/')
addpath('../NN/')
addpath('../util')

[ train_x, train_y, test_x, test_y ] = loadMNIST(  );

nn_findAvgPredict1(200,test_x,test_y);
nn_findAvgPredict2(200,test_x,test_y);
nn_findAvgPredict3(200,test_x,test_y);