fprintf('Correlated Probit Regression (CPR) Toy Example.\n\n')
addpath('./CPR/');								% load CPP files

%%%%% Generate Toy Data %%%%% 
fprintf('Generating Data...\n')
rng(141);										% seed
dim = 50;
N_train = 150;
N_test = 50;

% sparse ground truth
not_zero = 5;					
w_true = zeros(dim,1);
w_true(1:not_zero) = (10 + 3*randn(not_zero,1)).*ones(not_zero,1);

% generate data according to the probit model
N = N_train + N_test;
X = ones(dim,N) - 2*rand(dim,N); 				% generate uniformly distributed data points
Sigma = 0.1*((wishrnd(eye(N),N) + 0.2*eye(N))); % random cov matrix + diagonal noise
error = mvnrnd(zeros(N,1) , Sigma)'; 			% correlated error according to probit model
Y = sign( X'*w_true + error); 					% assign labels

% split data into training and test set
X_train = X(:,1:N_train);
Y_train = Y(1:N_train);
X_test = X(:,N_train+1:N);
Y_test = Y(N_train+1:N);
Sigma_train = Sigma(1:N_train,1:N_train);


%%%%% Training %%%%%
fprintf('Training...\n')
reg_lambda = 0.2; 								% reg constant for l1-norm regularizer 
cpr_setup(reg_lambda);
w = cpr_train(X_train, Y_train, Sigma_train, zeros(dim,1));


%%%%% Prediction %%%%%
fprintf('Prediction...\n')
[Y_predict, confidences] = cpr_predict(X, 1:N_train, N_train+1:N, Y_train, Sigma, w);



%%%%% Evaluation %%%%%
% accucary of prediction
accuracy = sum(Y_test ==Y_predict)/N_test;
fprintf('Accuracy CPR: %.2f\n', accuracy)

% prediction with true underlying weight vector
Y_oracle = sign(X_test'*w_true);
accuracy = sum(Y_test ==Y_oracle)/N_test;
fprintf('Accuracy Oracle: %.2f\n', accuracy)