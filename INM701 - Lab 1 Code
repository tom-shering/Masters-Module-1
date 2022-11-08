#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 21:28:43 2022

@author: tom
"""

# QUESTIONS I HAVE: 
    
# 1. Importing a dataset without using 'os' is possible as seen below,
#      so why do we need to import and use 'os'? 
# 2. What's the difference between standard deviation and variance? 

# NOTES I HAVE: 

# 1. Make a function that calculates the standard deviation of a dataset.



import pandas as pd
import os 
import math

# Question 1A

def mult_square(size):
    count = 1
    while count <= size:
        count2 = 1
        while count2 <= size:
            txt = str(count * count2).rjust(3)
            print(txt, end = ' ')
            count2 += 1
        print(' ')
        count += 1

mult_square(10)
print('\n')



# Question 1B

def mult_square2(size):
    count = 1
    while count <= size:
        count2 = 1
        while count2 <= size:
            if count * count2 % 2 == 0:
                txt = '0'.rjust(3)
                print(txt, end = ' ')
            else:
                txt = str(count * count2).rjust(3)
                print(txt, end = ' ')
            count2 += 1
        print(' ')
        count += 1

mult_square2(10)



# My own extension - building a 3D multiplication matrix:
    
print('\n My own 3d matrix: \n')

def mult_square(size):
    count = 1
    while count <= size:
        count2 = 1
        while count2 <= size:
            count3 = 1
            z_axis_ls = []
            while count3 <= size:
                num = count * count2 * count3
                z_axis_ls.append(num)
                count3 += 1
            txt = str(z_axis_ls).rjust(16)
            print(txt, end = ' ')
            count2 += 1
        print(' ')
        count += 1

mult_square(4)
print('\n')


# Part 2

# Importing a dataset without using 'os' is possible as seen below,
# so why do we need to import and use 'os'? 

# =============================================================================
# dataset = '/Users/tom/Documents/AI Course/INM701 - Intro/iris_dataset.csv'
# df = pd.read_csv(dataset)
# print(df[0:5])
# =============================================================================


# Showing that os is working:

print(os.getcwd())

# Importing a dataset using os:

path = '.'
filename_read = os.path.join(path, "/Users/tom/Documents/AI Course/INM701 - Intro/iris_dataset.csv")
df = pd.read_csv(filename_read)
print(df[0:5])

print('\n')

# Filtering a dataset using the 'loc' property: 

wide_petals_df = df.loc[df['petal_w'] > 1.0]
print(wide_petals_df)
print('\n')

# Sorting a dataframe column in ascending order: 
    
sepal_sort_df = df.sort_values(by = 'sepal_l')
print(sepal_sort_df)
print('\n')

# Apparently the sort_values method in pandas is great because unlike
# the python sort function, it can sort a dataframe and select by column, 
# as well as just having lots of nice changeable parameters. 

# Sorting a dataframe first by one column, and then by a second if there is 
# a tie:
# Notice the square brackets to keep both those column names under the 
# yoke of the 'by' parameter. 
    
sepal_w_doublesort_df = df.sort_values(by = ['sepal_w', 'sepal_l'] )
print(sepal_w_doublesort_df)
print('\n')

# Sorting in a descending order:
    
petal_sort_descend_df = df.sort_values(by = 'petal_l', ascending = False)
print(petal_sort_descend_df)
print('\n')

# There are several other sort_values parameters that can be utilised. 

# Save a dataframe to a new file: 

petal_sort_descend_df.to_csv('/Users/tom/Documents/AI Course/INM701 - Intro/petal_sort.csv')

# Calculate variance of a column: 
    
petal_w_variance = df['petal_w'].var()

print('The variance in petal width is: {}'.format(petal_w_variance))

petal_w_std = df['petal_w'].std()

print('The std in petal width is: {}'.format(petal_w_std))

# My own function for calculating variance: 
    
    
def calculate_standard_deviation(column):
    mean = sum(column) / len(column)
    sum_of_squared_differences = 0
    for num in column: 
        squared_difference = (num - mean) ** 2
        sum_of_squared_differences += squared_difference
    avg_of_sum_of_squared_diffs = sum_of_squared_differences / len(column)
    standard_deviation = math.sqrt(avg_of_sum_of_squared_diffs)
    return standard_deviation

petal_w_list = df['petal_w'].values.tolist()

print('My own calculated std is: {}'.format(calculate_standard_deviation(petal_w_list)))
        

































