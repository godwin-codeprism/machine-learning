# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 15:29:06 2017

@author: godwin
"""

#importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the datasets
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values

# splitting dataset into trainset and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=1/3,random_state=0)

# Fitting linear regression method to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)
 
# Predicting the Test set resutls
y_pred = regressor.predict(X_test)

# Visualising the Traning set results
plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("Salary vs Experience (Training Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# Visualising the Test set results
plt.scatter(X_test,Y_test,color="red")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("Salary vs Experience (Training Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()