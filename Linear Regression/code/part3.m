%THIS CODE FILE IS USED TO COMPUTE COEFFICIENTS FOR LINEAR REGRESSION (USING CLOSED FORM
%EXPRESSION SOLUTION WITH GLOBAL LEAST SQUARES ESTIMATE) BUT RELIABLE TRAIN WITH CROSS VALIDATION.

clear all;  % remove all open variables in work-space
close all;  % close all previous figures

%Parsing x06Simple.csv and extracting X and Y
filename = 'x06Simple.csv';
datafile = 'part2data.mat';

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
    
    %Read in the Feature Values
    X = csvread(filename, 1, 1);
    
    %Remove Last Column
    X(:, end) = [];
    
    %Read in Y
    Y = csvread(filename, 1, end, [1, end, size(X, 1), end]);
    
    %Close the file
    fclose(fid);
    
    %Save Data File
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

%Least Squares Estimate Stored for each fold (Visualization)
SE =[];

%Number of Folds = 5 here
folds = 5;

%STRATEGY - Round Off Fold Size, take lesser samples in last fold
fold_size = ceil(size(Xinput, 1)/folds);

%Set aside Training and Testing Data (for each fold iteration)
for i = 1:folds
   
    %The actual Fold for testing in iteration i
    k = ((i-1)*fold_size)+1; 
    test_ids = k:(k+fold_size-1);
    
    %Special Case Handling for Last Fold (i=folds)
    if(i==folds)
        test_ids = k:size(Xinput, 1);
    end
    
    % Everything Else is Training (Xinput - Test)
    train_ids = setdiff(1:size(Xinput,1), test_ids);
    
    %Extract Data using computed ID's
    Xtrain = Xinput(train_ids, :);
    Ytrain = Yinput(train_ids, :);

    Xtest = Xinput(test_ids, :);
    Ytest = Yinput(test_ids, :);

    %Standardize Training Data
    mu = mean(Xtrain);
    sdev = std(Xtrain);
    Xtrain = (Xtrain - mu)./sdev;

    %Add additional bias feature with value 1 to the Training data
    Xtrain = [ones(size(Xtrain,1), 1) Xtrain];

    %Compute Closed Form Parameter Estimates
    Xtrain_trans = Xtrain.';
    Theta = (inv(Xtrain_trans * Xtrain) * Xtrain_trans) * Ytrain;

    %Standardize Test Data with respect to Training Parameters (mean & std)
    Xtest = (Xtest - mu)./sdev;

    %Add additional bias feature with value 1 to the Testing data
    Xtest = [ones(size(Xtest,1), 1) Xtest];

    %Estimated Test Values
    Yestimate = Xtest * Theta;

    %Compute Squared Error's
    N = size(Xtest,1);
    SE = [SE sum((Ytest - Yestimate).^2)];
    
end  

%Compute Mean Squared Error & Root Mean Squared Error
MSE = (1/len)*(sum(SE));
RMSE = sqrt(MSE);

%Print RMSE
disp('Overall Test RMSE:');
disp(RMSE);



