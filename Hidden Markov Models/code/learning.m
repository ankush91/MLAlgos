%THIS CODE FILE IS USED TO IMPLEMENT THE PARAMETER LEARNING PROBLEM FOR A HIDDEN
%MARKOV MODEL USING THE BAUM-WELCH EXPECTATION MAXIMIZATION ALGORITHM

%PROBLEM STATEMENT :- TO LEARN PARAMETERS OF A HMM TOWARDS IMPROVED 
%ACCURACY OF OBSERVED CRIMINAL LOCATION REPORTS OVER TIME

close all; % remove all open variables in work-space
clear all; % close all previous figures

%**********LEARNING PROBLEM***********

%Transmission Matrix for location transtions (initially Uniform distribution)
a = [0.5, 0.5; 0.5, 0.5];

%Emission Matrix for location observations
b = [0.4, 0.1, 0.5; 0.1, 0.5, 0.4];

%location matrix
l = {'LA', 'NY', 'NULL'};

%Initial Pi for Priors (Uniform distribution)
pi = [0.5; 0.5];

%Probability Matrix P
P = [1];

%loop condition flag
flag = 0;

%count variable 
count = 0;

%Observations matrix
O = { 'NULL', 'LA', 'LA', 'NULL', 'NY', 'NULL', 'NY', 'NY', 'NY', 'NULL', 'NY', 'NY', 'NY', 'NY', 'NY', 'NULL', 'NULL', 'LA', 'LA', 'NY'};

%****LOOP TILL EVALUATION PROBABILITY CHANGE < EPS OR ITERATIONS >= 100****
while flag<=0 && count < 100

%Matrix at
at = [];

%***FORWARD ESTIMATION***

%Computation for Time Step 1 
for i = 1: size(pi, 1)
    at = [at ; pi(i) * b(i, find(strcmp(l, O(1))))];
end

%Computations for Time Step 2:num_obs
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
P = [P, sum(at(:, size(at, 2)))];

%Initial Matrix beta 
beta = [];
beta = [1; 1];
b1 = [];

%***BACKWARDS PROCEDURE***

%Computation for Time Step Last:Time Step 2 (decrement of -1)
for t = (size(O,2)-1):-1: 1
    
    %matrix for each time step
    b1t = [];
    
    %for each state i
    for i = 1: size(pi, 1)
        sumB = 0;
        
        %calculate betaj(t+1) * aij(t) * b(j, t+1) for all j
        for j = 1:size(pi, 1)
            b1 =  beta(j, 1) * a(i, j) * b(j, find(strcmp(l, O(t+1))));
            sumB = sumB + b1; 
        end
        
        %store results - matrix for each time step
        b1t = [b1t; sumB];
    end
    
    %copy to beta - matrix for all time steps
    beta = [b1t , beta];
end

%*********MAXIMIZATION**********

%****GAMMA CALCULATION****
gamma = [];

%Computation for Time Step 1:num_obs
for t = 1: size(O, 2)
    
    %matrix for each time step
    gammasub = [];
    
    %normalization constant for time step t
    sumG = 0;
    
    %calculate sum of deltaj(t) * betaj(t) for all j - store in sumG
     for j = 1:size(pi, 1)
        sumG = sumG + (at(j, t)*beta(j, t));
    end
    
   %for each state i 
   for i = 1:size(pi, 1)
       
       %calculate deltai(t) * betai(t) normalized by sumG
       gammai = (at(i, t) * beta(i, t))/sumG;
       
       %store results - matrix for each time step
       gammasub = [gammasub; gammai];
   end
   
   %copy to gamma - matrix for all time steps
   gamma = [gamma, gammasub];
end

%****EPSILON CALCULATION****
epsilon = [];

%Computation for Time Step 1:num_obs-1
for t = 1: size(O, 2)-1
    
   %matrix for each time step and each state 
   epsilon_sub = [];
   
   %normalization constant for each step t
   sumE = 0;
   
   %calculate sum of deltak(t) * betak(t) for all k - store in sumE
   for k = 1:size(pi, 1)
            sumE = sumE + at(k, t) * beta(k, t);
   end
   
   %for each state i
   for i = 1:size(pi, 1)
       
       epsiloni = []; 
       
       %calculate deltai(t) * aij * betaj(t+1) * bj(t+1) for all j
       for j = 1:size(pi, 1)
            epsilonij = at(i, t) * a(i, j) * beta(j, t+1) * b(j, find(strcmp(l, O(t+1))));
            epsiloni = [epsiloni, epsilonij/sumE];
       end
       
       %store results - matrix for each time step
       epsilon_sub = [epsilon_sub ; epsiloni];
   end
   
   %copy to epsilon - matrix for all time steps
   epsilon = [epsilon, epsilon_sub];

end

%*****UPDATE PARAMETERS****

%***update priors - pi***
pi = gamma(:, 1);

%***update transition matrix a***

%for each state i
for i = 1:size(pi, 1)
    
    %calculate normalizing constant for Time step t=1:num_obs-1
    normai = sum(gamma(i, :), 2) - gamma(i, end);
    
    %for all i to j
    for j = 1:size(pi, 1)
        sumaij = 0;
        
        %calculate sum(epsilonij(t)) for i,j and for Time step t=1:num_obs-1
        for t= 1:size(O, 2)-1
            sumaij = sumaij+ epsilon(i, (t-1)*2+j);
        end
        
        %store sum/normalizing_constant for i,j in a(i, j)
        a(i, j) = sumaij/normai;
    end    
end

%***update emission matrix b***

%for each state i
for i = 1:size(pi, 1)
    
    %calculate normalizing constant for Time step t=1:num_obs
    normbi = sum(gamma(i, :));
    
    %for all i, j (j here is observation value)
    for j = 1:size(l, 2)
        sumbij = 0;
        
        %calculate sum(gamma(i, t)) for all observations where t==j
        for t= 1:size(O, 2)
                if find(strcmp(O(t), l))==j
                    sumbij = sumbij+ gamma(i, t);
                end
        end
        
        %store sum/normalizing_constant for i,j in b(i, j)
        b(i, j) = sumbij/normbi;
    end    
end

%****TERMINATION CONDITION****
if(abs(P(1, size(P, 2)) - P(1, size(P, 2)-1)) < eps)
    flag = 1;
end

%increment count
count = count+1;

end

%****GRAPH VISUALIZATION OF PROBABILITY VALUES**** - till termination
P_final = P(2:size(P, 2));

hold off;
figure(1);
plot(1:count, P_final, 'b');
xlim([0 size(P_final, 2)]);
hold on;

xlabel('Iteration');
ylabel('Evaluation');

