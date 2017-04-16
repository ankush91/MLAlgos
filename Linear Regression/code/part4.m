%THIS CODE FILE IS USED TO COMPUTE COEFFICIENTS FOR LINEAR REGRESSION (USING CLOSED FORM
%EXPRESSION SOLUTION WITH **LOCALLY WEIGHTED** LEAST SQUARES ESTIMATE)

clear all; % remove all open variables in work-space
close all; % close all previous figures

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

%Standardize Test Data with respect to Training Parameters (mean & std)
Xtest = (Xtest - mu)./sdev;

%Add additional bias feature with value 1 to the Testing X data
Xtest = [ones(size(Xtest,1), 1) Xtest];

%Parameter k
k = 1;

Yestimate = [];

%For Each Query Point - Test data point (Train w.r.t Test data)
for i =1:size(Xtest, 1)
    
    Beta = [];
    
    %For Each Training Sample - Calculate beta function: e^(-d(a,b)/k^2)
    for j =1:size(Xtrain, 1)
        expression = ( -sum( abs(Xtest(i, :) - Xtrain(j, :)) ) ) / (k.^2); 
        Beta = [Beta exp(expression)];
    end
    
    %Calculate Weights Wx and Wy
    Wx = diag(Beta) * Xtrain;
    Wy = diag(Beta) * Ytrain;
    
    %Calculate Theta 
    Theta = ( inv(Wx.' * Wx) * Wx.') * Wy;
    
    %Predict Y and Add to list of Predicted Values
    Yestimate = [Yestimate; (Xtest(i, :)*Theta)];
end

%Compute Mean Squared Error and Root Mean Squared Error
N = size(Ytest, 1);
MSE = (1/N)*( sum((Ytest - Yestimate).^2) );
RMSE = sqrt(MSE);

%Print RMSE - Root Mean Squared Error
disp('RMSE Test:');
disp(RMSE);
