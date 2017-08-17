#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 19:57:02 2017

K - MEANS

@author: Ilaria
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import mall dataset with pandas
dataset = pd.read_csv('Mall_Customers.csv')

X = dataset.iloc[:,[3,4]].values

#find the optimal number of cluster with the elbow method
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, 
                    init = 'k-means++',
                    max_iter= 300,
                    n_init = 10,
                    random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_) 
    
#plot 
plt.plot(range(1,11),wcss)
plt.title('The elbow method for K-means')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#apply k-mean with the optimal #of clusters
kmeans = KMeans(n_clusters = 5, 
                    init = 'k-means++',
                    max_iter= 300,
                    n_init = 10,
                    random_state = 0)
#get a list of data points with the corrispondent cluster
y_kmeans = kmeans.fit_predict(X)

#plot the cluster results
plt.scatter(X[y_kmeans==0,0], X[y_kmeans==0,1], s=100, c='red', label ='Careful')
plt.scatter(X[y_kmeans==1,0], X[y_kmeans==1,1], s=100, c='blue', label ='Standard')
plt.scatter(X[y_kmeans==2,0], X[y_kmeans==2,1], s=100, c='green', label ='Target')
plt.scatter(X[y_kmeans==3,0], X[y_kmeans==3,1], s=100, c='magenta', label ='Careless')
plt.scatter(X[y_kmeans==4,0], X[y_kmeans==4,1], s=100, c='cyan', label ='Sensible')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s= 300, c='yellow', 
            label = 'centroid')
plt.title('Cluster of clients')
plt.xlabel('Annual salary K$')
plt.ylabel('Spending score')
plt.legend()
plt.show()



