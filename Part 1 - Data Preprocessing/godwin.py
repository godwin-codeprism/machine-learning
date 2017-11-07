# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import datasets
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,3].values

# Filling the missing values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy='mean',axis = 0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

#Pre processing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_x = LabelEncoder()
X[:,0] = labelEncoder_x.fit_transform(X[:,0])

oneHotEncode = OneHotEncoder(categorical_features=[0])
X = oneHotEncode.fit_transform(X).toarray()

labelEncoder_y = LabelEncoder()
Y = labelEncoder_y.fit_transform(Y)

# Splitting data into train set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state=0)

#Feature Scaling - bring both the variables into same scale
from sklearn.preprocessing import StandardScaler
x_SC = StandardScaler()
X_train = x_SC.fit_transform(X_train)
X_test = x_SC.transform(X_test)
