# Machine Learning Algorithms: Dynamic Implementations

This repository is part of the Implementations for Various Machine Algorithms using Given Assignment Specifications for Course Work in CS 613 (Machine Learning) Graduate Course work at Drexel Univeristy. The Implementations are in Matlab and need Matlab Version r2016b to run

## Basic Theory, PCA, K-means
This folder contains solutions to basic theory questions, relevant implementations for Principal Component Analysis and K-means (EM algorithm) on the 768 data points (8 features) diabetes.csv dataset.

## Linear Regression
This folder contains various implementations of Linear Regression such as the Closed form Solution (Global Least Squared Error), the Closed form Solution (with Cross-Validation), the Closed form Solution (Locally Weighted Linear Regression) and the Batch Gradient Descent Algorithm. The dataset used is 44 tuples (2 features - Age and Temperature of Water) used to predict Length of Fish.

## Classification - Naive Bayes & Multi-Class SVM
This folder contains implementations for the Naive Bayes Algorithm and Multi-Class SVM (using MATLAB's fitcsvm function to compare the ONE-VS-ONE and ONE-VS-ALL approach). The email SPAM dataset comprising of 4601 datapoints (57 continuous valued features) is used for the Naive Bayes classification. The Cartioocgraphy dataset is used for the Multi-Class SVM problem and comprises of 2126 datapoints (21 features) and the objective is to determine fetal class codes given observations.

## Artificial Neural Networks
This folder contains implementations for BINARY & MULTI-CLASS Artificial Neural Networks (3 Layers | 20 Node Hidden Layer) using the Batch Gradient Descent Algorithm. 2-D Visualization for the Precision vs Recall graph is done for the Binary Case (in part2.m). The email SPAM dataset comprising of 4601 datapoints (57 continuous valued features) is used for the Binary classification case. The Cartioocgraphy dataset comprising of 2126 datapoints (21 features) is used for the Multi-Class Artificial Neural Network problem and the objective is to determine fetal class codes given observations.

## Hidden Markov Models
This folder contains implementations for the Evaluation and Learning Problem of Hidden Markov Models (first order). The Evaluation problem is solved using the recursive Forward Algorithm and the Learning problem is solved using the Baum Welch Expectation Maximization Algorithm. The data is a sample of reports of successive location observations for a travelling criminal, the hidden states are the actual locations and state tranistions are the transitions between locations. The objective is to determine the probability of observations given the model (evaluation.m) & learn the parameters of the model to fit the observations (learning.m).
