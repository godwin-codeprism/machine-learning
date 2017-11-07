# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 17:56:04 2017

@author: godwin
"""

#importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the datasets
dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values