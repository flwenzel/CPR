%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Florian Wenzel
% 2016
%
% Setup hyper parameters and other properties for the CPR training procedure.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function cpr_setup(lambda)
    global reg_lambda1;     % reg constant for l1-regularizer
    global reg_lambda2;     % reg constant before l2-term in ADMM scheme (has no effect on the solution)
    global opt_tol;         % convergence crit
    global stepsize;        % stepsize of newton's method
    global new_steps_ADMM;  % number newton steps within ADMM loop
    global verbose;         % show information every *verbose* step

	reg_lambda1 = lambda; 
	reg_lambda2 = 0.5; 
	opt_tol = 0.1;
	stepsize = 0.9;
	new_steps_ADMM = 1;
	verbose = 10;
end