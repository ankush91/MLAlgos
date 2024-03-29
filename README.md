# Machine Learning Algorithms from Scratch: Matlab
* Implementations of various Machine Algorithms using assignment specifications as part of graduate course work in CS 613 (Machine Learning), taught at Drexel Univeristy in Fall 2016
* The Implementations are in Matlab (tested on version r2016b)

## [Eigen Theory, PCA, K-means](./Eigen%20Theory%2C%20PCA%2C%20K-means/)
* Contains basic Eignen theory questions and their solutions
* Implementatations of Principal Component Analysis and K-means (using EM algorithm) on 768 data points and 8 features of the `diabetes.csv` dataset

## [Linear Regression](./Linear%20Regression/)
* Various implementations of Linear Regression such as the Closed-form Solution (Global Least Squared Error), the Closed-form Solution with Cross-Validation, the Closed-form Solution using Locally Weighted Linear Regression and the Batch Gradient Descent Algorithm.
* Dataset is from 44 samples (2 features - Age and Temperature of Water) used to predict Length of Fish

## [Classification - Naive Bayes & Multi-Class SVM](./Classification%20-%20Naive%20Bayes%20%26%20Multi-Class%20SVM/)
* Implementations of the Naive Bayes Algorithm and Multi-Class SVM (using MATLAB's `fitcsvm` function to compare the One-VS-One and One-VS-All approach)
* An email Spam dataset comprising of 4601 samples and 57 continuous valued features is used in the Naive Bayes classification task
* The Cartioocgraphy dataset is used for Multi-Class SVM and comprises of 2126 samples and 21 features). The objective is to determine fetal class codes given observations

## [Artificial Neural Networks](./Artificial%20Neural%20Networks/)
* Implementations of Binary & Multi-Class Artificial Neural Networks (3 Layers | 20 Nodes per Hidden Layer) using the Batch Gradient Descent Algorithm
* An email Spam dataset comprising of 4601 samples and 57 continuous valued features is used for the Binary classification case
* 2D Visualization of Precision vs Recall is also carried out for the Binary classification case (in part2.m)
* The Cartioocgraphy dataset comprising of 2126 samples and 21 features is used for the Multi-Class Artificial Neural Network problem. The objective is to determine fetal class codes given observations

## [Hidden Markov Models](./Hidden%20Markov%20Models/)
* Implements the Evaluation and Learning tasks of 1st order Hidden Markov Models 
* The Evaluation problem is solved using the recursive Forward Algorithm
* The Learning problem is solved using the Baum Welch Expectation Maximization Algorithm 
* The dataset is a sample of successive location observations for a travelling criminal. The hidden states are the actual locations and state tranistions are the transitions between locations. The objective is to determine the probability of observations given the model (evaluation.m) & learn the parameters of the model to fit these observations (learning.m)
