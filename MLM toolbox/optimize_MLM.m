function [optParam, Ecv] =  optimize_MLM(data, param, bias, lambda, type, nFolds, option)

N = size(data.x, 1);
if(nFolds == N),
    CVO = cvpartition(N, 'Leaveout');
else
    CVO = cvpartition(N, 'kfold', nFolds);
end

amseValues = zeros(nFolds, length(param));
for i = 1: nFolds,
    learnPoints.x = data.x(training(CVO, i), :);
    learnPoints.y = data.y(training(CVO, i), :);
    testData.x = data.x(test(CVO, i), :);
    testData.y = data.y(test(CVO, i), :);
    for j = 1: length(param),
        if(param(j) <= 1),
            K = round(param(j)*size(learnPoints.x, 1));
        else
            K = param(j);        
        end
        refPoints.x = learnPoints.x(1:K, :);
        refPoints.y = learnPoints.y(1:K, :);
        [model] = train_MLM(refPoints, learnPoints, bias, lambda);    
        [amseValues(i, j)] = AMSE(testData, model, type, option);
    end
end    
Ecv = mean(amseValues);
[~, indice] = min(Ecv);
optParam = param(indice);
