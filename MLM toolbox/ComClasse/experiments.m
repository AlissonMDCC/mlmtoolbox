

trials = 10;
LP_MSE = zeros(trials, 1);
LSQ_MSE = zeros(trials, 1);

DB_NAME = '04-computer_activity';
load(['../uci-regression-data/' DB_NAME]);



for i=1:trials
   data.x = xtrain{i}; data.y = ytrain{i};
   MLM = MinimalLearningMachine(data, data);
   yhLP = MLM.predict(xtest{i}, 'LP');
   yhLSQ = MLM.predict(xtest{i}, 'lsqnonlin');
   
   LP_MSE(i) = mean( (ytest{i} - yhLP).^2 );
   LSQ_MSE(i) = mean( (ytest{i} - yhLSQ).^2 );
end

save(['results_' DB_NAME], 'LP_MSE', 'LSQ_MSE');