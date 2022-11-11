#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 19:44:15 2022

@author: tom
"""

import os 
import pandas as pd 
import numpy as np

from sklearn import datasets 

cancer = datasets.load_breast_cancer()

X = cancer.data
y = cancer.target

print(X.shape)
print(y.shape)


df = pd.DataFrame(data=np.c_[cancer.data,cancer.target],columns=np.append(cancer.feature_names,'target'))

print(df.head())

print(y[0:5])
print(y[399:404])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# Data is often scaled before learning, and StandardScaler is one class that does this.
#This transforms data so that has a normal distribution with a mean
#of 0 and a standard deviation of 1.
#This means that all features are distributed across the same distribution, 
# hence the learning algorithm isn't biased by the magnitude of values.
from sklearn.preprocessing import StandardScaler


from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the iris dataset from scikit-learn
cancer = datasets.load_breast_cancer()

X = cancer.data
y = cancer.target

# Split the data
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Perceptron, with its training parameters
ppn = Perceptron(max_iter=60,tol=0.001,eta0=1)

# Train the model
ppn.fit(X_train,y_train)

# Make predication
y_pred = ppn.predict(X_test)

# Evaluate accuracy
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

# The trained model is a set of parameters defining a decision surface (a hyperplane) and we can see what these are:
print(ppn.coef_)


# Counting the labels of the target to confirm that the result isn't just
# a balance of the labels 
unique, counts = np.unique(y_test, return_counts=True)
dict(zip(unique, counts))

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

diagnosis = cancer.target_names

#python function, defined here to plot confusion matrices
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
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
plt.figure()
plot_confusion_matrix(cm, diagnosis, title='')

#graphical plots of confusion matrix using method above
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized, diagnosis, title='Normalized confusion matrix')
plt.show()


# Another way of making a confusion matrix with Skikit-learn is shown below. 

from sklearn.metrics import ConfusionMatrixDisplay

sc = StandardScaler()

sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

print(X_train[0])
print(X_train_std[0])

ppn = Perceptron(max_iter=40,tol=0.001,eta0=1)

# Train the model
ppn.fit(X_train_std,y_train)

# Make predication
y_pred = ppn.predict(X_test_std)

cm = confusion_matrix(y_test, y_pred)

# Evaluate accuracy
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ppn.classes_)
disp.plot()
plt.show()


# Decision Trees 

from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

iris = datasets.load_iris()

print(iris.data[:5])
print(iris.target[:5])

print(iris.feature_names)

X = iris.data
y = iris.target
variety = iris.target_names

print(X.shape)
print(y.shape)

print(X[:5])
print(y[:5])
print(y[145:150])

# Split the data:
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Model is trained with the fit() method:

decision_tree = DecisionTreeClassifier(criterion = 'entropy')

decision_tree.fit(X_train, y_train)

# Evaluate the trained data: 

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

print(cm)

plt.figure()
plot_confusion_matrix(cm, variety, title='')


# Note: It seems important to always lable data and target as X and y, things
# don't work as well otherwise and some inbuilt methods call these 
# by name. 

# Display a different confusion matrix: 
    
# from sklearn.metrics import ConfusionMatrixDisplay

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=variety)
disp.plot()
plt.show()

# Using 5-fold cross validation

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

plt.figure()
plot_confusion_matrix(cm_5_fold, variety, title='')





























