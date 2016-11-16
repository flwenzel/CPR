%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Prediction - Correlated Probit Regression (CPR)
% Florian Wenzel
% 2016
%
% Predict labels for test points. This method requires the test *and* training data since the joint
% likelihood between training and test points is evaluated.
%
% Inputs 
% X_data: 		(n_train+n_test)xd matrix of training points and test points
% idxTrain:		indexes of training points
% idxTest:		indexes of test points
% Y_train:		training labels
% Sigma:		kernel matrix of all points (training points and test points)
% w_train:		weight vector obtained by cpr_train
%
% Outputs
% predictions:	predicted labels for test points
% confidences:	predicted probability the test points belong to class 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [predictions, confidences] = cpr_predict(X_data, idxTrain, idxTest, Y_train, Sigma, w_train)
	% data
	N_train = length(idxTrain); 
	N_test = length(idxTest);
	label = [-1, 1];

	% storing
	highestLikelihood = 0;
	bestY = 0;
	predictions = zeros(N_test,1);
	confidences = zeros(N_test,1);

	% EP
	lowerB = zeros(N_train+1,1);
	upperB = 1e6 * ones(N_train+1,1);

	% compute joint likelihood of training labels and test label
	% (for each test point separably)
	for i = 1:N_test
		idxTestPoint = idxTest(i);
		idx = [idxTrain, idxTestPoint];

		X_all = X_data(:,idx);
		Sigma_all = Sigma(idx,idx);


		for j=1:2
			Y_test = label(j);
			Y_all = [Y_train; Y_test];

			% transform Sigma
			Sigma_trans = diag(Y_all) * Sigma_all * diag(Y_all);

			% compute mu
			mu = Y_all .* (X_all' * w_train);

			% evaluate log likelihood
			[logZEP, mu_hat, Sigma_hat, cunningham_iter, error,  mu_last_iteration, sigma_last_iteration] = axisepmgp_warmStart(mu,Sigma_trans,lowerB,upperB,0, 1e-5, 0,0,0);

			% select label with higher log likelihood
			if(j==1)
				highestLikelihood = logZEP;
				bestY = Y_test;
				counterConfidences(i) = highestLikelihood;
			elseif(logZEP > highestLikelihood)
				highestLikelihood = logZEP;
				bestY = Y_test;
			elseif(logZEP < highestLikelihood)
				counterConfidences(i) = logZEP;
			end
		end

		predictions(i) = bestY; 
		confidences(i) = exp(highestLikelihood);
	end

	% normalize probabilities
	counterConfidences = exp(counterConfidences)';
	evidence = (counterConfidences + confidences);
	counterConfidences = counterConfidences./evidence;
	confidences = confidences./evidence;

	% report confidences wrt to label "+1"
	for i=1:N_test
		if predictions(i)==-1
			confidences(i) = 1-confidences(i);              
		end             
	end


end