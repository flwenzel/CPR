%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Training - Correlated Probit Regression (CPR)
% Florian Wenzel
% 2016
%
% Training of the CPR model.
%
% Inputs
% X:        nxd matrix of training points
% Y:        vector of training labels
% Sigma:    kernel matrix of training points
% w0:       initial value of w for warm start
%
% Outputs
% w:         weight vector of CPR model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function w = cpr_train(X, Y, Sigma, w0)
    % get global parameters
    global reg_lambda1;     % reg constant for l1-regularizer
    global reg_lambda2;     % reg constant for l2-regularizer
    global opt_tol;         % convergence crit
    global stepsize;        % stepsize of newton's method
    global new_steps_ADMM;  % number newton steps within ADMM loop
    global verbose;         % show information every *verbose* step

    % data
    n_points = size(X,2);
    dim = size(X,1);

    % transform Sigma
    Sigma = diag(Y)*Sigma*diag(Y);

    % precalculate values
    invSigma = inv(Sigma);
    grad_last_term = invSigma * diag(Y) * X';
    X = X.*repmat(Y',dim,1);
    K = X'*X;

    % parameters for EP
    lowerB = zeros(n_points,1);
    upperB = 1e6 * ones(n_points,1);
    mu_warmStart = 0;
    sigma_warmStart = 0;

    % init
    steps = 0;
    grad = 2*opt_tol; % arbitrary init s.t. convergence crit is not fulfilled
    newton_grad = 1;
    z = zeros(dim,1) + 0.01;
    eta = zeros(dim,1);

    % warm start
    w = w0;


    while(norm(grad)>opt_tol || norm(w-z,1)/dim>0.01) % convergence criterium
        steps = steps  + 1;

        % warm start for EP
        if(steps  == 1)
            use_warmStart = 0;
        else
            use_warmStart = 1;
        end
        
        %%% ADMM Scheme %%%

        % w Step (with inner Newton optimization loop)
        newton_steps = 0;
        for new_steps = 1:new_steps_ADMM
            newton_steps = newton_steps + 1;
           
            % get moments by EP
            mu = X'*w;
            [logZEP, mu_hat, Sigma_hat, EP_steps,  error,  warmstart1, warmstart2] = axisepmgp_warmStart(mu,Sigma,lowerB,upperB,0, 1e-5, mu_warmStart, sigma_warmStart, use_warmStart);
            
            % store moments for warm start
            mu_warmStart = warmstart1;
            sigma_warmStart = warmstart2;
            
            % compute gradient
            dmu = mu - mu_hat;
            grad = ( dmu' * grad_last_term )' + reg_lambda2 * (w - z + eta);

            % compute inverses hessian (using woodbury matrix identity for speedup)
            hessian_mu_inv = -inv( (Sigma_hat - dmu*dmu') * invSigma - eye(n_points) + 1e-5*eye(n_points) ) * Sigma;
            
            % compute inverse hessian times gradient
            x = X'*grad;
            x = ( hessian_mu_inv + 1/reg_lambda2*K ) \ x;
            x =  -1/reg_lambda2^2 * X*x;
            y = 1/reg_lambda2*grad;
            newton_grad = y + x;

            % Newton Step
            w = w - stepsize * newton_grad;


        end

        % z Step
        soft_threshold_k = reg_lambda1/reg_lambda2;
        soft_threshold_a = w + eta;
        z = soft_threshold(soft_threshold_k, soft_threshold_a);
        
        % eta step
        eta = eta + w - z;


        % print some values
        if ~(verbose==-1) && mod(steps,verbose) == 0
            fprintf('step = %d',steps);
            fprintf('   grad = %.3f', norm(grad));
            fprintf('   newton_grad = %.3f', norm(newton_grad));
            fprintf('\n')
        end
    end



end