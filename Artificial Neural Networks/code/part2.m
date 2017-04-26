%THIS CODE FILE IS USED TO IMPLEMENT AN ARTIFICAL NEURAL NETWORK FOR BINARY
%CLASSIFICATION USING BATCH GRADIENT DESCENT
% &
%TO VISUALIZE THE PRECISION RECALL TRADEOFF GRAPH BY VARYING THE OUTPUT THRESHOLD

clear all; % remove all open variables in work-space
close all; % close all previous figures

%Parsing spambase.data and extracting X and Y
filename = 'spambase.data';
datafile = 'ann1.mat';

%Load Data File if it exists
if(exist(datafile, 'file'))
    load(datafile);
else
    %Else Open the .data file
    fid = fopen(filename);
    
    %Check if the File Exists
    if(fid < 0)
        disp('file not found');
        return;
    end
    
    %Extract X - Read in the Feature Values
    X = importdata(filename, ',', 0);
    
    %Extract Y - Read in the Output Values
    Y = X(:, size(X, 2));
    
    %Clear Last Column of X
    X(:, size(X, 2)) = [];
    
    %Close the file
    fclose(fid);
    
    %Save Data Values
    save(datafile, 'X', 'Y');
end

%Set Initial Seed
rng(0);

%Size of X
len = size(X, 1);

%Random Permutation of the Indices - upto len 
R = randperm(len);

%Take in Input-Output Pairs in Random Order
for i=1:len
    Xinput(i, :) = X(R(i), 1:end);
    Yinput(i, :) = Y(R(i), 1:end);
end

%Set aside (2/3rd) Training and (1/3rd) Testing Data
limit = ceil(len*0.667);
next = limit+1;

Xtrain = Xinput(1:limit, :);
Ytrain = Yinput(1:limit, :);

Xtest = Xinput(next:end, :);
Ytest = Yinput(next:end, :);

%Standardize Training Data
mu = mean(Xtrain);
sdev = std(Xtrain);
Xtrain = (Xtrain - mu)./sdev;

%Add additional bias feature with value 1 to the Training X data
Xtrain = [ones(size(Xtrain,1), 1) Xtrain];

%Number of Hidden Layer Nodes num_h = 20
num_h = 20;

%N (size of training set) and learning rate alpha
N = size(Xtrain, 1);
alpha = 0.5;

%Set Initial Seed
rng(0);

%Initialize Beta - Weights for Hidden Layer Nodes 
beta = rand(size(Xtrain, 2), num_h);

%Interval = [+1, -1]
beta = (1+1).*beta - 1;

%Initialize Theta - Weights for Output Layer Nodes
theta = rand(num_h, 1);

%Interval = [+1, -1]
theta = (1+1).*theta - 1;

%error_count
train_error = [];

%Weight for Batch Update
factor = (alpha/N);

%OUTER LOOP (Forward-Backword Propagation for 1000 iterations)
for iterations = 1:1000
    
    %***********FORWARD PROPAGATION***********
    %Compute Hidden Layer Node Values
    h = 1./(1+ exp(-1.*(Xtrain*beta)));
    
    %Compute Output Layer Node Values
    o = 1./(1+ exp(-1.*(h*theta)));
    
   %Classify using o values AND Store Label in oc
    oc = [];
    for i =1:N
        if(o(i) > 0.5)
            oc = [oc; 1];
        else
            oc = [oc; 0];
        end
    end

    %Add Training Correct Count
    count = 0;
    for i = 1:N
        if(Ytrain(i)==oc(i))
            count = count + 1;  
        end
    end
    
    % Caclulate Train Error
    train_error = [train_error; (1-(count/N))];
    
    %**********BACKWARD PROPAGATION***********
    
    %************OUTER-LAYER**********
    %Compute loss at output node
    delta1 = [];
    delta1 = (Ytrain - o) .* o .*(1 - o);

    %Update parameter Theta by avg of weighted losses
    theta = theta + factor.*(delta1.' * h).';

    %**********HIDDEN-LAYER************
    %Compute loss at hidden node
    delta2 = [];
    prod = (theta * delta1.').';
    delta2 = prod .* h .* (1-h);

    %Update parameter Beta by avg of weighted losses
    beta = beta + factor.*(delta2.'*Xtrain).';

end

%*****************TESTING*****************

%Standardize Test Data using Training Parameters - mean and std
Xtest = (Xtest - mu)./sdev;
Xtest = [ones(size(Xtest,1), 1) Xtest];

%Initialize Variables for Precision-Recall Curve
inc =0.0;
precision = [];
recall = [];

%*****Loop for Precision VS Recall*******
while inc <= 1

%Compute Hidden Layer Node Values
h = 1./(1+ exp(-1.*(Xtest*beta)));
    
%Compute Output Layer Node Values
o = 1./(1+ exp(-1.*(h*theta)));
    
%Classify using o values and Threshold (inc)
 oc = [];
 for i =1:size(Xtest, 1)
    if(o(i) > inc)
        oc = [oc; 1];
    else
        oc = [oc; 0];
    end
 end

%Output Count Labels for TP, FP, TN, FN
label = [];

%For Loop to Count Labels
for i =1:size(Ytest, 1)
    
    if(oc(i) == Ytest(i) && oc(i)== 1)
        label = [label; 'P'];
    elseif(oc(i) == Ytest(i) && oc(i) == 0)
        label = [label; 'N'];
    elseif(oc(i) ~= Ytest(i) && oc(i) == 1)
        label = [label; 'V'];  
    elseif(oc(i) ~= Ytest(i) && oc(i) == 0)
        label = [label; 'W'];   
   
    end
end

%Compute TP, TN, FP & FN using Label Counts
TP_count = sum(label=='P');
TN_count = sum(label=='N');
FP_count = sum(label=='V');
FN_count = sum(label=='W');

%----Compute Statistics for Precision & Recall----
p = TP_count/(TP_count+FP_count);
r = TP_count/(TP_count+FN_count);

%Edge Case for NAN values (divide by 0)
if(TP_count == 0 && FP_count == 0)
   r = 0;  
   p = 1;
end

%Add Values to Vector
precision = [precision; p];
recall = [recall; r];
 
 %increase increment
 inc = inc + 0.1;
 
end
 
 %**********Visualization of Graph for Precision VS Recall**********
hold off;
figure(1);
plot(precision, recall, '-bo');
hold on;

xlabel('Precision');
ylabel('Recall');

axis([0 1 0 1]);



