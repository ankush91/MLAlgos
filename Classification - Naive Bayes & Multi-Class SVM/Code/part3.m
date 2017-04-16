%THIS CODE FILE IS USED TO PERFORM MULTI-CLASS SVM USING THE ONE-VS-ALL AND
%ONE-VS-ONE APPROACH

clear all; % remove all open variables in work-space
close all; % close all previous figures

%Parsing CTG.csv and extracting X and Y
filename = 'CTG.csv';
datafile = 'SVM.mat';

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
    
    %Extract All
    X = csvread(filename, 2, 1);
    
    %Extract Y - Output Values
    Y = X(:, size(X, 2));
    
    %Clear Last And Second to Last Column of X - Extract features
    X(:, size(X, 2)) = [];
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

%Take in Input-Ouput Pairs in Random Order
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

%Standardize X Training Data
mu = mean(Xtrain);
sdev = std(Xtrain);
Xtrain = (Xtrain - mu)./sdev;

%Standardize X Test Data
Xtest = (Xtest - mu)./sdev;

%-----ONE VS ALL APPROACH-----

%find number of unique classes
n_values = unique(Ytrain, 'rows');

%For loop (Training and Testing) for ALL CLASSES
for i =1 : size(n_values, 1)
    
    %Declare All as zeros
    Y_train_i = zeros(size(Ytrain, 1), 1); 
    
    %iteration i class 1
    Y_train_i(find(Ytrain==n_values(i)), :) = 1;    
    
    %Train classifier - fitcsvm library
    model = fitcsvm(Xtrain, Y_train_i);
    
    %Predict using Trained Classifier
    if i == 1
        %predict using trained SVM model
        Ypredict = predict(model, Xtest);
    else
        Ypredict = [Ypredict predict(model, Xtest)];
    end
    
end  

%Decide Classification
Y_label = [];

%Seed Random Number Generator
rng(0);
    
for i = 1:size(Ypredict, 1)
    
    %Find Column Numbers of Row where Y-classification = 1
    Y_column = find(Ypredict(i, :)==1);
    
    %If Multiple Columns Returned
    if(size(Y_column, 2) > 1)
       
     %Assign one of them as Label
        R = randperm(size(Y_column, 2));
        Y_label = [Y_label; n_values(R(1))];
     
    %Else if 1 classification is returned  
    elseif(size(Y_column) == 1)
        %Assign the single index
        Y_label = [Y_label; n_values(Y_column)]; 
    
    %Else - Assign any of them at random
    else
        R = randperm(size(n_values, 2));
        Y_label = [Y_label; n_values(R(1))];
    
    end
end

%Accuracy - ONE VS ALL APPROACH
accuracy_1 = sum(Y_label == Ytest)/size(Ytest, 1);

%-----ONE VS ONE APPROACH-----

Ypredict_i_j = [];

%For loop (Training and Testing) - Choose i classifier as label 1
for i =1 : size(n_values, 1)   
    
    %Choose j classifier as label 0
    for j = i+1: size(n_values, 1)
     
    %k(k-1)/2 Classifiers        
            %iteration i-Class i OR iteration j-Class j
            Ytrain_i_j = Ytrain(find(Ytrain==n_values(i)), :); 
            Ytrain_i_j = [Ytrain_i_j ; Ytrain(find(Ytrain==n_values(j)), :)];
    
            Xtrain_i_j = Xtrain(find(Ytrain==n_values(i)), :);
            Xtrain_i_j = [Xtrain_i_j ; Xtrain(find(Ytrain==n_values(j)), :)];
    
            %Train classifier - fitcsvm library function
            model = fitcsvm(Xtrain_i_j, Ytrain_i_j);
    
            %Predict using Trained Classifier - model
            Ypredict_i_j = [Ypredict_i_j, predict(model, Xtest)];
        
    end
    
end  

%Decide Classification
Y_label_2 = [];
Y_count = [];

%Find Count of Each Label for Each Row
for i = 1:size(n_values, 1)
    
    Y_count = [Y_count, sum(Ypredict_i_j == n_values(i), 2)];

end

%Seed Random Number Generator
rng(0);

%Find Column Numbers (Labels) where Count = MAXIMUM
  for i=1:size(Y_count, 1)
    
    %Find Maximum label(column number) count in a row
    num = max(Y_count(i, :));
    column_2 = find(Y_count(i, :) == num);
    
        %Assign Maxmimum Count Label (Column Number) as the predicted label
        if(size(column_2, 2) == 1)
            Y_label_2 = [Y_label_2; n_values(column_2)];
            
        %In case of tie, Assign Randomly out of the Maximum Labels as the predicted label    
        else
            R = randperm(size(column_2, 2));
            Y_label_2 = [Y_label_2; n_values(R(1,1))];
        end   
    end

%Accuracy - ONE VS ONE APPROACH
accuracy_2 = sum(Y_label_2 == Ytest)/size(Ytest, 1);

%Display BOTH Accuracies and Compare
disp('Accuracy for ONE VS ALL:');
disp(accuracy_1);

disp('Accuracy for ONE VS ONE:');
disp(accuracy_2);
    
    

