#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 19:57:02 2017

HIERARCHICAL CLUSTERING

@author: Ilaria
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import mall dataset with pandas
dataset = pd.read_csv('Mall_Customers.csv')

X = dataset.iloc[:,[3,4]].values

#find the optimal number of cluster using the dendrogram
#first print it
import scipy.cluster.hierarchy as sch
#ward min the variance between clusters
dendogram = sch.dendrogram(sch.linkage(X,method = 'ward'))
plt.title('Dendogram')
plt.xlabel ('Customers')
plt.ylabel('Euclidean distance')
plt.show()

#fitting hierarchical clustering to mall dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

#visualize the cluster - just to plot clusters in 2 dimensions
plt.scatter(X[y_hc==0,0], X[y_hc==0,1], s=100, c='red', label ='Careful')
plt.scatter(X[y_hc==1,0], X[y_hc==1,1], s=100, c='blue', label ='Standard')
plt.scatter(X[y_hc==2,0], X[y_hc==2,1], s=100, c='green', label ='Target')
plt.scatter(X[y_hc==3,0], X[y_hc==3,1], s=100, c='magenta', label ='Careless')
plt.scatter(X[y_hc==4,0], X[y_hc==4,1], s=100, c='cyan', label ='Sensible')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s= 300, c='yellow', 
            label = 'centroid')
plt.title('Cluster of clients')
plt.xlabel('Annual salary K$')
plt.ylabel('Spending score')
plt.legend()
plt.show()
