clear all

numberRepetitions = 10;
param = 0.1:0.1:1;
bias = 0;
lambda = 0; %10e-10;
type = 'fsolve';
nFolds = 10;
option = 'distances';

%names = {'18-wine.mat', '17-pima_indians_diabetes.mat'};
dataset = load('uci-classification-data/18-wine.mat');

%% pre-processing / 1-of-S output encoding scheme
labels = unique(dataset.ytrain{1});
code = zeros(length(labels), length(labels));
for j = 1: length(labels),
    code(j, j) = 1;
end
for i = 1: numberRepetitions,
    for j = length(labels):-1:1,
        ind = (dataset.ytrain{i} == labels(j));
        ind2 = (dataset.ytest{i} == labels(j));
        tam = length(find(ind==1));
        tam2 = length(find(ind2==1));
        dataset_ytrain{i}(ind, :) = repmat(code(j, :), tam, 1);
        dataset_ytest{i}(ind2, :) = repmat(code(j, :), tam2, 1);    
    end
end

%% basic methodology
for i = 1: numberRepetitions,
    data.x = dataset.xtrain{i};
    data.y = dataset_ytrain{i};

    [opt_parameter, Ecv{i}] = optimize_MLM(data, param, bias, lambda, type, nFolds, option);

    K = round(opt_parameter*size(data.x, 1));
    ind = randperm(size(data.x,1));
    refPoints.x = data.x(ind(1:K), :);
    refPoints.y = data.y(ind(1:K), :);

    [model] = train_MLM(refPoints, data, bias, lambda);

    testData.x = dataset.xtest{i};
    testData.y = dataset_ytest{i};
    [Yh, error] = test_MLM(model, testData, type);  
    
    for j = 1: length(testData.y),
        [~, index(j)] = max(Yh(j,:));
        [~, target(j)] = max(testData.y(j, :));
    end  
    rate = length(find(index == target))/length(testData.y);

    confusionMatrix = zeros(size(code, 1));
    for j = 1 : length(testData.y),
        confusionMatrix(index(j), target(j)) = confusionMatrix(index(j), target(j)) + 1;
    end
    rates(i) = rate;
    output{i} = Yh;
    confusionMatrices{i} = confusionMatrix;
end
mean(rates)
std(rates)
targets = dataset.ytest;
save('results_MLM_wine', 'rates', 'confusionMatrices', 'Ecv', 'output', 'targets');