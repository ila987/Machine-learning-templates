#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 18:50:11 2017

APRIORI

@author: Ilaria
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import mall dataset with pandas
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)

#import dataset as a list of list
transaction = []
for i in range (0,7501):
    transaction.append([str(dataset.values[i,j]) for j in range(0,20)])
  
#training a priori on dataset
from apyori import apriori
rules = apriori(transaction, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_lenght = 2)

#visualizing results
myResults = [list(x) for x in results]

