%THIS CODE IS USED TO PERFORM DIMENSIONALITY REDUCTION USING PRINCIPAL
%COMPONENT ANALYSIS

clear all; % remove all open variables in work-space
close all; % close all previous figures

%Parsing diabetes.csv and extracting X and Y
filename = 'diabetes.csv';
datafile = 'part2data.mat';

%Load Data File if it exists
if(exist(datafile, 'file'))
    load(datafile);
%Else Open the .csv file
else
    fid = fopen(filename);
    %Check if the File Exists
    if(fid < 0)
        disp('file not found');
        return;
    end
    
    %Read in the Feature Values
    X = csvread(filename, 0, 1);
    
    %Read in the Output Value
    Y = csvread(filename, 0, 0 , [0, 0, size(X, 1)-1, 0]);
    
    %Close the File
    fclose(fid);
    
    %Save Data for future Use
    save(datafile, 'X', 'Y');
end

%Standardize X values along column dimension
X = (X - mean(X))./std(X);

%Extract the Covariance Matrix & Compute Eigen Vector and Eigen Values
Cov_Matrix = cov(X);

[Eigvec, Eigval] = eig(Cov_Matrix);

%Extract Eigen Values
[Maxval, Maxind] = max(Eigval);

%Reduce data to 2 dimensions here
k = 2;

%Extract Eigen Vectors along i = 1..k (2 here) dimensions
for i = 1:k
     
    %Extract EigenValue of ith Variability (Descending)
    [Max_i, Maxind_i] = max(Maxval);
     Maxval(Maxind_i) = [];
     Eig_i = Eigvec(:, Maxind_i);
     
     %Store in EigW
     if (i == 1)
        EigW = Eig_i;
     else
         EigW = [EigW Eig_i];
     end     
end

%Project all Data Points onto K (2 here) dimensions
Z = X*(EigW);

%Plot Data points and Verify Result by Visualization
figure(1);
plot(Z(Y==-1,1), Z(Y==-1,2), 'xb');
hold on;
plot(Z(Y==1,1), Z(Y==1,2), 'or');
hold off;

title('PCA');
xlabel('x1');
ylabel('x2');





    
    