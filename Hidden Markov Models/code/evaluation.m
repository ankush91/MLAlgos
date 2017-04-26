%THIS CODE FILE IS USED TO IMPLEMENT THE EVALUATION PROBLEM FOR A HIDDEN
%MARKOV MODEL USING THE FORWARD/EVALUATION ALGORITHM

%PROBLEM STATEMENT :- TO EVALUATE LIKELIHOOD OF OBSERVED
%CRIMINAL LOCATION REPORTS OVER TIME

close all; % remove all open variables in work-space
clear all; % close all previous figures

%**********EVALUATION PROBLEM***********

%Transmission Matrix for location transtions (Uniform distribution)
a = [0.5, 0.5; 0.5, 0.5];

%Emission Matrix for location observations
b = [0.4, 0.1, 0.5; 0.1, 0.5, 0.4];

%location matrix
l = {'LA', 'NY', 'NULL'};

%Initial Pi for Priors (Uniform distribution)
pi = [0.5; 0.5];

%Observations matrix
O = { 'NULL', 'LA', 'LA', 'NULL', 'NY', 'NULL', 'NY', 'NY', 'NY', 'NULL', 'NY', 'NY', 'NY', 'NY', 'NY', 'NULL', 'NULL', 'LA', 'LA', 'NY'};

%Matrix to store estimates for all time steps
at = [];

%***FORWARD ALGORITHM***

%Computation for Time Step 1  - BASE STEP
for i = 1: size(pi, 1)
    at = [at ; pi(i) * b(i, find(strcmp(l, O(1))))];
end

%Computations for Time Step 2:num_obs - RECURSIVE STEP
for t = 2:size(O,2)
    
    %matrix for each time step
    at_sub = [];
    
    %for each state i
    for i = 1: size(pi, 1)
            sumA = 0;
            
            %calculate aj(t-1) * aji(t) for all j
            for j = 1:size(pi, 1)
                sumA = sumA + (at(j, t-1) * a(j, i));
            end
            
            %multiply with bik for obs k at time step t
            a1t = b(i, find(strcmp(l, O(t)))) * sumA;
            
            %store results to matrix for each time step
            at_sub = [at_sub ; a1t]; 
    end
    
    %copy to at - matrix for all time steps
    at = [at, at_sub];
end

%Evaluation Probability
P = sum(at(:, size(at, 2)));

%Display Probability 
disp(P);