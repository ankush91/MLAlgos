%THIS CODE FILE IS USED TO COMPUTE COEFFICIENTS OF LINEAR REGRESSION USING
%BATCH GRADIENT DESCENT

clear all; % remove all open variables in work-space
close all; % close all previous figures

%Parsing spambase.data and extracting X and Y
filename = 'spambase.data';
datafile = 'partNaiveBayes.mat';

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
    
    %Extract X - Read in features (Observations)
    X = importdata(filename, ',', 0);
    
    %Extract Y - Output Values
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

%Set Aside Spam and Non-Spam Data in Training Phase
Xtrain_spam = Xtrain(find(Ytrain==1), :);
Xtrain_not_spam = Xtrain(find(Ytrain==0), :);

%Estimate Priors (Use Proportions in Training Set as Estimators)
Prior_Xtrain_spam = size(Xtrain_spam, 1)/(size(Xtrain, 1));
Prior_Xtrain_not_spam = size(Xtrain_not_spam, 1)/(size(Xtrain, 1));

%Calculate Means of Features in each Class
mean_Xtrain_spam = mean(Xtrain_spam);
mean_Xtrain_not_spam = mean(Xtrain_not_spam);

%Calculate Standard Deviation of Features in each Class
std_Xtrain_spam = std(Xtrain_spam);
std_Xtrain_not_spam = std(Xtrain_not_spam);

%Standardize Xtest Values
Xtest_standardize = (Xtest - mu)./sdev;

%For each Test Sample Calculate Norm pdf's (Gaussian Distribution) using
%Training Set Mean and Std
p_x_spam = normpdf(Xtest_standardize, mean_Xtrain_spam, std_Xtrain_spam);
p_x_not_spam = normpdf(Xtest_standardize, mean_Xtrain_not_spam, std_Xtrain_not_spam);

%Calculate Maximum Likelihood Estimate for Each Test Sample in each class
% Naive Assumption - Complete Independence
MLE_spam = prod(p_x_spam, 2);
MLE_not_spam = prod(p_x_not_spam, 2);

%Multiply MLE with Prior's & Normalize
p_Ytest_estimate_spam = (Prior_Xtrain_spam .* MLE_spam);
p_Ytest_estimate_not_spam = (Prior_Xtrain_not_spam .* MLE_not_spam);

%Classify according to Probabilities
Ytest_predict = zeros(size(Ytest, 1), 1);

%If posterior for spam > not_spam then Spam else not_spam
for i =1:size(Ytest, 1)
    
    if(p_Ytest_estimate_spam(i) > p_Ytest_estimate_not_spam(i))
        Ytest_predict(i) = 1;
    else
        Ytest_predict(i) = 0;
    end
end

%Compare Ytest and Ypredict - Count Labels
label = [];

%Labels to Calculate TP, TN, FP & FN
for i =1:size(Ytest, 1)
    
    if(Ytest_predict(i) == Ytest(i) && Ytest_predict(i)== 1)
        label = [label; 'P'];
    elseif(Ytest_predict(i) == Ytest(i) && Ytest_predict(i) == 0)
        label = [label; 'N'];
    elseif(Ytest_predict(i) ~= Ytest(i) && Ytest_predict(i) == 1)
        label = [label; 'V'];  
    elseif(Ytest_predict(i) ~= Ytest(i) && Ytest_predict(i) == 0)
        label = [label; 'W'];   
   
    end
end

%Count, TP, TN, FP and FN
TP_count = sum(label=='P');
TN_count = sum(label=='N');
FP_count = sum(label=='V');
FN_count = sum(label=='W');

%----Compute Statistics----
%Precision, Recall, f-1 measure and Accuracy
precision = TP_count/(TP_count+FP_count);
recall = TP_count/(TP_count+FN_count);
f_measure = (2*precision*recall)/(precision+recall);
accuracy = (TP_count+TN_count)/(TP_count+TN_count+FP_count+FN_count);

%Display Results
disp('Precision:');
disp(precision);

disp('Recall:');
disp(recall);

disp('F-measure');
disp(f_measure);

disp('Accuracy');
disp(accuracy);



