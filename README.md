# Wine Quality Prediction

This repository contains a machine-learning model for predicting wine quality based on its physicochemical properties. The project leverages various data preprocessing and model-building techniques, including logistic regression, decision trees, random forests, and more.

# Project Overview

The goal of this project is to build a machine-learning model that predicts the quality of wine based on its chemical attributes. The dataset contains various features such as acidity, residual sugar, pH, and alcohol content. By using this dataset, we aim to predict wine quality on a scale from 0 to 10. The project includes :
  * Text preprocessing
  * Feature extraction
  * Model building and evaluation
    
We use different machine learning models to classify the sentiment and compare their performance.
# Dataset
The dataset used in this project is the Wine Quality Dataset provided in . The dataset contains the following columns:
  * fixed acidity
  * volatile acidity
  * citric acid
  * residual sugar
  * chlorides
  * free sulfur dioxide
  * total sulfur dioxide
  * density
  * pH
  * sulphates
  * alcohol
  * quality (Target variable)

# Installation
To get started, you need to install the required libraries. You can do this by running the following command:
![Screenshot 2024-11-29 201944](https://github.com/user-attachments/assets/6a455740-cf1e-414e-85d3-6e375b3b3786)
  * pandas: For data manipulation
  * numpy: For numerical operations
  * matplotlib & seaborn: For data visualization
  * scikit-learn: For machine learning models and metrics

# Importing Libraries
You import the following libraries for data manipulation, statistical analysis, and visualization:

  * import pandas as pd
  * import statsmodels.formula.api as smf
  * import statsmodels.stats.multicomp as multi
  * import scipy.stats
  * import numpy as np
  * import seaborn as sns
  * import matplotlib.pyplot as plt

# Data Exploration & Preprocessing
you explore the dataset to understand the relationships between different features and the target variable (quality).

  * You check for missing values, distributions of each feature, and basic summary statistics.
  *  Visualize correlations using a heatmap to understand which features are strongly correlated     with the wine quality.
  * Execute all the codes for exploratory data analysis and data wrangling without missing a single line.
  * Run the codes related to individual functions and their plots consecutively as given in the .ipynb file. These include:
      *  Covariance matrix
      * Adding category quality
      * Exploring statistical interactions of low, medium, and high quality.
  * Perform data exploration, frequency distributions, and all the required plots for data visualization one by one, following the given sequence.
    
# Model Building

  * Import the required libraries such as sklearn, numpy, matplotlib, scipy, and operator, including all necessary extensions. Execute the single code block defining all the models to be executed further.

  * Run the individual functions for the following models in any order to achieve the desired output:
    * Logistic Regression model to predict the quality of wine.
    * Decision Tree model, which is a simple yet interpretable model.
    * K-NN model for classification.
    * Naive Bayes classifier.
    * Random Forest is an ensemble method that combines multiple decision trees to improve accuracy.
    * Linear Regression to validate the accuracy that it doesn't work on categorical data.

# Conclusion

This project demonstrates how to use machine learning models to predict the quality of wine based on its physicochemical properties. We evaluated several models, and Random Forest performed the best in terms of accuracy.
