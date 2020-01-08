# Sparkify-predicting-customer-churn
Predicting customer churn using Spark

## Motivation 
Attracting new customers to your innovative and lucrative business might be easy in some cases. But the real effort thereafter is in retaining those customers in the long run. In this project, a fictional music app Sparkify aim to predict those customer who are willing to leave the service using Apache Spark as tool. Our aim is to predict whether a user is going to churn ie. give up on the streaming service completely based on their interactions with the app.

![image](music_streaming_cmpnies.png)

## Files
This repository contains following files: 
- mini_sparkify_event_data.json: a small subset of the original data used to train and test our model;
- Sparkify.ipynb: Jupyter notebook for analysis
- Sparkify.html: a HTML version of the Jupyter notebook 

## Requirements

The following important libraries are required - 
* pandas
* matplotlib
* seaborn
* pyspark
* statsmodels

## Approach and Results
In order to predict customer churn rates first I transformed data into numerical values and also created new features. 
In the second step I implemented following algorithms enabled by Spark:

1. Logistic Regression
2. Decision Tree Classifier
3. Gradient Boosting

In the first run the algorithms were specified without parameter tuning in order to improve their time performance.
The results of the first run were:
1. The logistic regression model has a accuracy of: 0.81, and F1 score of:0.44, using 334.29 seconds.
2. The decision tree model has a accuracy of: 0.83, and F1 score of: 0.57, using 60.08 seconds.
3. The gradient boosted trees model has a accuracy of: 0.85, and F1 score of: 0.6, using 294.64 seconds.

After the first run I chose gradient boosting for further tuning. After parameter tuning gradient boosting scored even better on this dataset with F1 score over 0.83. As this data set is contains a tiny subset (128MB) of the full dataset available (12GB), hence further training on full dataset might improve the prediction. 
<<<<<<< HEAD

=======
>>>>>>> 5987c4934777dcea61819a94a5753f6c3780af2c
