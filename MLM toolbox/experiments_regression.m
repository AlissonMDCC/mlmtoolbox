
clear all; clc;
rep = 10;
param = 0.5:0.5:0.5;
bias = 0;
lambda = 0; %10e-10;
type = 'ppl';
nFolds = 10;
option='distances';
names = {'09-servo'};
data_name = names{1};
%load the dataset
path = 'uci-regression-data/';
dataset_name = strcat(path, data_name,'.mat');
dataset = load(dataset_name);


for i = 1: rep,
    data.x = dataset.xtrain{i};
    data.y = dataset.ytrain{i};

     [opt_parameter, Ecv{i}] = optimize_MLM(data, param, bias, lambda, type, nFolds, option);

    K = round(opt_parameter*size(data.x, 1));
    ind = randperm(size(data.x,1));
    refPoints.x = data.x(ind(1:K), :);
    refPoints.y = data.y(ind(1:K), :);

    [model] = train_MLM(refPoints, data, bias, lambda);

    testData.x = dataset.xtest{i};
    testData.y = dataset.ytest{i};
    [Yh, error] = test_MLM(model, testData, type);               
    final_error(i) = mean(error.^2)
end

MSE = mean(final_error)
stdev = std(final_error)  


file1 = strcat('results_MLM_', data_name, '_', num2str(bias), '_', type, '.mat');
save(file1, 'MSE', 'stdev', 'param', 'Ecv', 'final_error');

