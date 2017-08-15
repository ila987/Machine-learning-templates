#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 13:50:26 2017

Polynomial regression
Use both a linear and a poliinomial regression to show the different results


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

#fitting linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#fitting polinomial regression 
from sklearn.preprocessing import PolynomialFeatures
poly_reg =  PolynomialFeatures(degree = 4)
X_poli = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poli, y)

#visualize linear regression results
plt.scatter(X, y, color= 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Reality vs Bluff(linear regression)')
plt.xlabel('Position label')
plt.ylabel('Salary')
plt.show()


#visualize polinomial regression results
plt.scatter(X, y, color= 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Reality vs Bluff(polynomial regression)')
plt.xlabel('Position label')
plt.ylabel('Salary')
plt.show()


#Predicting a new result with Linear Regression
lin_reg.predict(6.5)


#Predicting a new result with Polinomial regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))










