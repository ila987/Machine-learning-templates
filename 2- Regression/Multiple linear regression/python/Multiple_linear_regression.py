#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 11:08:12 2017

Multiple linear regression
Import dataset and encode categorical variables, fit multiple linear regression
find best model using backward elimination

@author: Ilaria
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#Encode categorical values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder_X = LabelEncoder()
X[:,3] = label_encoder_X.fit_transform(X[:,3])
#We add as many columns as dummy variable, to avoid order issue with the encoding of the variable
oneHotencoder = OneHotEncoder(categorical_features=[3])
X= oneHotencoder.fit_transform(X).toarray()

#Avoiding the Dummy variable trap 
#without taking all dummy variables but - 1
X = X[:, 1:]
            
                
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Fitting Multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting test set results
y_pred = regressor.predict(X_test)


#build optimal model using Backward Elimination
import statsmodels.formula.api as sm
#add constant column with all 1 to have a b0
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
#fit model with all predictors
X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
#let's remove the variable with the highest  pvalue
X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
#let's remove the variable with the highest  pvalue
X_opt = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
#let's remove the variable with the highest  pvalue
X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
#let's remove the variable with the highest  pvalue
X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()



















