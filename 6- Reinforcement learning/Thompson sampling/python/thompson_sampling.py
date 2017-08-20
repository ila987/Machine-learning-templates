#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 18:50:11 2017

THOMPSON SAMPLING

@author: Ilaria
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')


#implementing UCB
N = 10000
d = 10
numbers_of_rewards_1 = [0]*d
numbers_of_rewards_0 = [0]*d
ads_selected = []    
total_reward = 0             
#all users                  
for n in range(0,N):
    ad = 0
    max_random = 0
    #all version of the ads
    for i in range(0,d):
        random_beta = random.betavariate(numbers_of_rewards_1[i]+1, numbers_of_rewards_0[i]+1); 
        
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if(reward ==1 ):
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] +1
    else:
        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] +1
    total_reward = total_reward + reward
    
    
#visualizing the results
plt.hist(ads_selected)
plt.title('Histogram of ads selection')
plt.xlabel('ads')
plt.ylabel('Number of times each ad was selected')    
    
    