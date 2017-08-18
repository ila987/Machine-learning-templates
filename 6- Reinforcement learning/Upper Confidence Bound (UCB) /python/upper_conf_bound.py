#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 18:50:11 2017

UPPER CONFIDENCE BOUND

@author: Ilaria
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')


#implementing UCB
N = 10000
d = 10
numbers_of_selection = [0] * d
sums_of_rewards = [0] * d  
ads_selected = []    
total_reward = 0             
#all users                  
for n in range(0,N):
    ad = 0
    max_upper_bound = 0
    #all version of the ads
    for i in range(0,d):
        if(numbers_of_selection[i]> 0):
            average_reward= sums_of_rewards[i]/numbers_of_selection[i]
            delta_i = math.sqrt(3/2*math.log(n+1)/numbers_of_selection[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selection[ad] = numbers_of_selection[ad] + 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward
    
    
#visualizing the results
plt.hist(ads_selected)
plt.title('Histogram of ads selection')
plt.xlabel('ads')
plt.ylabel('Number of times each ad was selected')    
    
    