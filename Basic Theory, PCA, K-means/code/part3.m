%THIS CODE FILE IS USED FOR PERFORMING K-MEANS (EM algorithm) WITH NUMBER OF CLUSTERS = 2. 

clear all; % remove all open variables in work-space
close all; % close all previous figures

%Parsing diabetes.csv and extracting X and Y
filename = 'diabetes.csv';
datafile = 'part3data.mat';

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

%Standardize X values along row dimension
X = (X - mean(X, 1))./std(X);

%Extract the 6th and 7th feature of the standardized data set
Xinput = X(1:end, 6:7);

%Change the number of clusters k here 
k = 2;

%Set Initial Seed
rng(0);

%Generation of 2 random instances for initial mean values
for i = 1:k
        
        %Generate Random Number
        row = int32 ( randi ([1, size(Xinput,1)], 1, 1) ) ; 
        
        %Extract row from Xinput
        u_next = Xinput(row, 1:end); 
        if i == 1
            u = u_next;
        else
            u = [u; u_next];
        end
        
end

%Declare another vector as the size of the means vector
u_prev = zeros(size(u));

%Loop Initialization, flags and Y
i = 1;
Y = [size(Xinput,1)];
condition = true;

%Plot the seed values
plotting_seed(Xinput, u);

%Outer while loop condition and infinite loop contingency
while condition && (i <= 100)
    
    %Assign mean as previous calculation
    u_prev = u;
    
    %For loop for all X value's
    for j1 = 1:size(Xinput, 1)
        
        Distance = Inf;
        temp = Inf;
        
        %Match against all clusters with L1 distance
        for cluster_number = 1:size(u,1)
            
            temp = sqrt( sum( (Xinput(j1, :) - u(cluster_number, :)).^2) );
            
            %Assign Cluster number to X and store in Y
            if (temp < Distance)
                Y(j1, 1) = cluster_number;
                Distance = temp;         
            end
     
        end      
        
    end   
     
    %Compute new mean values
    for j2 = 1:size(u,1)
        Zind = find(Y(:, 1) == j2);
        Z = Xinput(Zind, :);
        u(j2, :) = mean(Z, 1);
    end  
    
    count = 0;
    %Check condition for loop termination -Sum of Manhattan Distance < epsilon
    Sum_L1 = [];
    for j3 = 1:size(u,1)
        manhattan = sum( abs(u(j3, :) - u_prev(j3, :)) );
        
        if(j3 == 1)
            Sum_L1 = manhattan;
        else
            Sum_L1 = [Sum_L1 manhattan];
        end
    end
    
    if(sum(Sum_L1(1, :)) < eps)
        condition = false;
    end
    
    %plotting the cluster graphs
    if(condition==false || i==1)
    %Plotting Function    
    plotting(Xinput, Y, i, u(1, :), u(2, :));
    end
    %increment i
    i=i+1;
    
end 

%Function to Plot X of different clusters
function [] = plotting(X1, Y1, i, u1, u2)
figure(i+1);
plot(X1(Y1==1,2), X1(Y1==1,1), 'xr');
hold on;
plot(X1(Y1==2,2), X1(Y1==2,1), 'xb');
hold on;
plot(u1(1, 2),u1(1, 1), 'or');
hold on;
plot(u2(1, 2),u2(1, 1), 'ob');
hold off;

out = ['Iteration' num2str(i)];
title(out);
xlabel('7th feature')
ylabel('6th feature')

end

%Function to Plot Initial seed
function [] = plotting_seed(X1, u)
figure(1);
plot(X1(:, 2),X1(:, 1),'xr');
hold on;
plot(u(:, 2), u(:, 1), 'ob');

title('Initial seeds');
xlabel('7th feature')
ylabel('6th feature')

end




        
        
            
        



