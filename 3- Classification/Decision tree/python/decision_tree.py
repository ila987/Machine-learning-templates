#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 20:33:25 2017

DECISION TREE CLASSIFIER

@author: Ilaria
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#fitting decision tree classifier to the training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion= 'entropy', 
                                    random_state = 0)
classifier.fit(X_train, y_train)


#predict test set result
y_pred = classifier.predict(X_test)

#making the confusion matrix
from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test, y_pred)

#visualizing the training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min() - 1, stop = X_set[:,0].max() + 1, step=0.01), 
                     np.arange(start = X_set[:,1].min() - 1, stop = X_set[:,1].max() + 1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), 
             alpha = 0.75, cmap= ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X2.max())
plt.ylim(X2.min(), X1.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j, 0], X_set[y_set==j, 1], 
                c= ListedColormap(('red', 'green'))(i), label= j)

plt.title('Decision tree classifier (training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.show()


#visualizing the test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min() - 1, stop = X_set[:,0].max() + 1, step=0.01), 
                     np.arange(start = X_set[:,1].min() - 1, stop = X_set[:,1].max() + 1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), 
             alpha = 0.75, cmap= ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X2.max())
plt.ylim(X2.min(), X1.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j, 0], X_set[y_set==j, 1], 
                c= ListedColormap(('red', 'green'))(i), label= j)

plt.title('Decision tree classifier (test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.show()

# Plotting the tree
# In the terminal enter: pip install pydot2
from sklearn import tree
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydot
dot_data = StringIO()
tree.export_graphviz(classifier,
                     out_file = dot_data,
                     feature_names = ['Age', 'Estimated Salary'],
                     class_names = ['Yes', 'No'],
                     filled = True,
                     rounded = True,
                     special_characters = True)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())


