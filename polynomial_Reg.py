# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 11:48:11 2018

@author: Manit
"""

"""Building a model-
   1) All in
   2) Backward Elimination
   3) Forward Elimination
   4)Bidirectional Elimination
   5) Score Comparision
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv('Position_Salaries.csv')
X=data.iloc[:,1:2].values
Y=data.iloc[:,2].values

from sklearn.linear_model import LinearRegression
lin_reg= LinearRegression()
lin_reg.fit(X,Y)

#Polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=5)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly,Y)
lin_reg_2=LinearRegression()
lin_reg_2.fit(X_poly,Y)
plt.scatter(X,Y,color='red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)))
plt.show()

lin_reg.predict(7)
lin_reg_2.predict(poly_reg.fit_transform(7))

X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y,color='red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)))
plt.show()
