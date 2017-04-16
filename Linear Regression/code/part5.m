%THIS CODE FILE IS USED TO COMPUTE COEFFICIENTS OF LINEAR REGRESSION USING
%BATCH GRADIENT DESCENT

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
    Y = csvread(filename, 1, 3, [1, 3, size(X, 1), 3]);
    
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

%Set Initial Seed again(For Parameter Values)
rng(0);

%Initialize 'n' Parameter Values in Range, R[-1, 1]
n = size(Xtrain, 2);
R = [-1 1];
Theta = rand(n, 1)*range(R) + min(R);

%Initialize loop conditions, flags, learning rate alpha and vectors for RMSE
iterations = 1000000;
condition = true;
count = 1;
alpha = 0.01;
RMSE_train = [];
RMSE_test = [];

%Size of Training and Testing Data
Ntrain = size(Xtrain, 1);
Ntest = size(Xtest, 1);

%While condition is true and count < max_iterations
while condition && (count<iterations)
   
    %Assign Theta Theta_prev for stop condition below
    Theta_prev = Theta;
   
    %For Each Paramater
    for j=1:size(Theta) 
        
        %Update (Sum) for each parameter
        update = 0;
        
        %For Each Training Sample
        for i=1:size(Xtrain,1)
            expression = ( (Xtrain(i, :) * Theta) - Ytrain(i, :) ) * Xtrain(i, j);
            update = expression+update;    
        end
        
        %Update as per Batch Gradient Descent
        Theta(j) = Theta(j) - ((alpha/Ntrain) * update);    
    end
    
    %Mean Squared Error and Root Mean Squared Error for Training and
    %Testing Data (current Iteration)
    MSE_test= (1/Ntest)*( sum((Ytest - Xtest*Theta).^2) );
    RMSE_test = [RMSE_test sqrt(MSE_test)];
    
    MSE_train = (1/Ntrain)*( sum((Ytrain - Xtrain*Theta).^2) );
    RMSE_train = [RMSE_train sqrt(MSE_train)];
    
    %Check Loop Termination Condition
    manhattan = abs(Theta - Theta_prev);
    
    %Set Condition as False if Values Less than EPS
    if(sum(manhattan) < eps)
        condition = false;
    end
    
    %increment i
    count = count+1;
end

%Final RMSE (for Testing & Training)
RMSE_test_final = RMSE_test(end);
RMSE_train_final = RMSE_train(end);

%Print Final Model Parameter and RMSE Values
disp('Theta');
disp(Theta);
disp('Final RMSE Test:');
disp(RMSE_test_final);
disp('Final RMSE Train:');
disp(RMSE_train_final);

%Plot Entire Progress of RMSE for Training and Testing Data
figure(1);
plot(1:(count-1), RMSE_train, 'r');
hold on;
plot(1:(count-1), RMSE_test, 'b');
hold off;

title('Gradient Descent Progress');
xlabel('Iteration');
ylabel('RMSE of Data');
legend('Training Error', 'Testing Error');