#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 13:43:41 2017

Random forest regression

@author: Ilaria
"""


# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#we don't apply training and test set because we don't have enough data
# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
"""

#fitting Random forest regression model to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(X,y)


#Predicting a new result with Random forest regression regression
y_pred = regressor.predict(6.5)


#visualize Random forest regression results(for high resolution and smoother curves)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color= 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Reality vs Bluff(decision tree regression)')
plt.xlabel('Position label')
plt.ylabel('Salary')
plt.show()











