#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 08:16:17 2022

@author: tom
"""

import os 
import pandas as pd 
import numpy as np

from sklearn import datasets 
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

wine = datasets.load_wine()

X = wine.data
y = wine.target

print(X.shape)
print(y.shape)

wine_df = pd.DataFrame(data = X)

# Use iloc to specify which rows and columns you want to display. 
print(wine_df.iloc[ : 5, : 5])
print()

# Finding the mean of the first column: 
print("Mean of first column: " + str(wine_df[0].mean()))
print()


variety = wine.target_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Decision Tree Analysis:
    
print("Decision Tree Analysis: ")

decision_tree = DecisionTreeClassifier(criterion = 'entropy')

decision_tree.fit(X_train, y_train)

y_pred = decision_tree.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print('Accuracy: %.2f' % accuracy)

def plot_confusion_matrix(cm, names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(fraction=0.05)
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
cm = confusion_matrix(y_test, y_pred)
print()

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=variety)
disp.plot()
plt.show()

# 5-fold Cross Validation: 
    
print("5-fold Cross Validation: ")
    
kf = KFold(5,shuffle=True)

fold = 1
# The data is split five ways, for each fold, the 
# Perceptron is trained, tested and evaluated for accuracy
for train_index, validate_index in kf.split(X,y):
    decision_tree.fit(X[train_index], y[train_index])
    y_test = y[validate_index]
    y_pred = decision_tree.predict(X[validate_index])
    #print(y_test)
    #print(y_pred)
    #print(f"Fold #{fold}, Training Size: {len(trainDF)}, Validation Size: {len(validateDF)}")
    print(f"Fold #{fold}, Training Size: {len(X[train_index])}, Validation Size: {len(X[validate_index])}")
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    fold += 1 

cm_5_fold = confusion_matrix(y_test, y_pred)

print(cm_5_fold)
print()

plt.figure()
plot_confusion_matrix(cm_5_fold, variety, title='')   

# After using decision tree and 5-fold cross validation, the decision tree 
# method gives a higher degree of accuracy (0.92 as opposed to 0.89).

# I will now use 10-fold cross validation to see if this increases the 
# accuracy of this method. 

print("10-fold Cross Validation: ")

kf_10 = KFold(10,shuffle=True)

fold = 1
# The data is split five ways, for each fold, the 
# Perceptron is trained, tested and evaluated for accuracy
for train_index, validate_index in kf_10.split(X,y):
    decision_tree.fit(X[train_index], y[train_index])
    y_test = y[validate_index]
    y_pred = decision_tree.predict(X[validate_index])
    print(y_test)
    print(y_pred)
    #print(f"Fold #{fold}, Training Size: {len(trainDF)}, Validation Size: {len(validateDF)}")
    print(f"Fold #{fold}, Training Size: {len(X[train_index])}, Validation Size: {len(X[validate_index])}")
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    fold += 1 

cm_10_fold = confusion_matrix(y_test, y_pred)

print(cm_10_fold)

plt.figure()
plot_confusion_matrix(cm_5_fold, variety, title='') 

# Note: In 10-fold cross validation, there are less data points on the 
# confusion matrix. Does that mean it was unable to predict less data rows? 

# Note: Is the overall accuracy of a k-fold validation method the mean of 
# the accuracies for each fold? If so, is there an automated way to do this? 

# Note: We need to apply the Perceptron model to the iris dataset but 
# I'm not sure what this enatils, I thought it was included in the above? 






























# End. 



