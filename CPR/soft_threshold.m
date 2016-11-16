%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Florian Wenzel
% 2016
%
% Soft treshold function used for ADMM.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [z] = soft_treshold(k, a)
    z = zeros(size(a));
    idx1 = a>k;
    idx2 = a<-k;
    z(idx1) = a(idx1) - k;
    z(idx2) = a(idx2) + k;
end