# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 09:32:05 2020

@author: Aksha
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics

"""=================== Read CSV Data ==========="""

df = pd.read_csv(r"G:\PYTHON\ml\regression-stock-prediction-master\goog.csv")
#print(df.head())

"""================ Split the Dependent and Indipendent Features==== """
X = df[["Open", "High", "Low", "Volume"]].values
y = df['Close'].values

"""================ Train the model ===================="""
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

regressor = LinearRegression()
regressor.fit(x_train,y_train)

#regressor.coef_

"""Plot the data"""
import matplotlib.pyplot as plt
plt.scatter(df[['Open']], y, color= 'black', label= 'Data')


y_pred = regressor.predict(x_test)
print(y_pred)
 
result = pd.DataFrame({'Actual':y_test.flatten(),'Predicted':y_pred.flatten()})
result.head(25)

""" measure the error of model"""
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))