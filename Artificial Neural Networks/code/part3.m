%THIS CODE FILE IS USED TO IMPLEMENT AN ARTIFICAL NUERAL NETWORK FOR
%MULTI-CLASS CLASSIFICATION USING BATCH GRADIENT DESCENT.

%PROBLEM STATEMENT: TO DETERMINE FETAL STATE CLASS CODE GIVEN AN
%OBSERVATION

clear all; % remove all open variables in work-space
close all; % close all previous figures

%Parsing CTG.csv and extracting X and Y
filename = 'CTG.csv';
datafile = 'ann2.mat';

%Load Data File if it exists
if(exist(datafile, 'file'))
    load(datafile);
else
    %Else Open the .csv file
    fid = fopen(filename);
    
    %Check if the File Exists
    if(fid < 0)
        disp('file not found');
        return;
    end
    
    %Extract All Columns
    X = csvread(filename, 2, 1);
    
    %Extract Y - Extract Output Values
    Y = X(:, size(X, 2));
    
    %Clear Last And Second to Last Column of X - Extract Features
    X(:, size(X, 2)) = [];
    X(:, size(X, 2)) = [];
    
    %Close the file
    fclose(fid);
    
    %Save Data Values
    save(datafile, 'X', 'Y');
end


%Set Initial Seed
rng(0);

%Size of X - upto len
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

%Number of Output Layer Nodes o_count = Number of unique values (Classes)
o_count = size(unique(Y), 1);

%Output Theshold Y_exp
Y_exp = 0.5;

%N (size of training set) and learning rate alpha
N = size(Xtrain, 1);
alpha = 0.5;

%Initialize Beta
beta = rand(size(Xtrain, 2), num_h);

%Interval = [+1, -1]
beta = (1+1).*beta - 1;

%Initialize Theta
theta = rand(num_h, o_count);

%Interval = [+1, -1]
theta = (1+1).*theta - 1;

%error_count
train_error = [];

%Weight for Batch Update
factor = (alpha/N);

%Number of Unique Values in Y
val = unique(Y);

%Compute Initial Error at output nodes
loss = [];
for m = 1 : N
  for n = 1 :o_count
      if(Ytrain(m) == val(n))
          loss(m, n) = 1;
      else
          loss(m, n) = 0;
      end
  end
end

%OUTER LOOP (Forward-Backword Propagation for 1000 iterations)
for iterations = 1:1000
    
    %***********FORWARD PROPAGATION***********
    %Compute Hidden Layer Node Values
    h = 1./(1+ exp(-1.*(Xtrain*beta)));
    
    %Compute Output Layer Values
    o = 1./(1+ exp(-1.*(h*theta)));
    
   %Classify using o values
    oc = [];
    [v id] = max(o, [], 2);
    oc = val(id);

    %Add Training Correct Count
    count = 0;
    for i = 1:N
        if(Ytrain(i)==oc(i))
            count = count + 1;  
        end
    end
    
    %Calculate Train Error
    train_error = [train_error; (1-(count/N))];
    
    %**********BACKWARD PROPAGATION***********
    
    %************OUTER-LAYER**********
    %Compute loss at output node        
    delta1 = [];
    for i = 1:o_count
        delta1 = [delta1, (loss(:, i) - o(:, i)) .* o(:, i) .*(1 - o(:, i))];
    end
    
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

%Standardize Test Data using Training Parameters - mean & std
Xtest = (Xtest - mu)./sdev;
Xtest = [ones(size(Xtest,1), 1) Xtest];

 %Compute Hidden Layer Node Values
 h = 1./(1+ exp(-1.*(Xtest*beta)));
    
 %Compute Output Layer Values
 o = 1./(1+ exp(-1.*(h*theta)));
    
 %Classify using maximum o probability values
  oc = [];
  [v id] = max(o, [], 2);
  oc = val(id);
  
  %Add Test Correct Count
  count = 0;
  for i = 1:size(Xtest, 1)
      if(Ytest(i)==oc(i))
          count = count + 1;  
      end
  end
 
 %Display Test Error
 disp('Test Error');
 test_error = 1 - count/size(Xtest, 1);
 disp(test_error);

 %**********Graph Visualization for training error vs iteration**********
hold off;
figure(1);
plot(1:iterations, train_error, 'b');
hold on;

xlabel('Iteration');
ylabel('Training Error');




