
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

#Load the dataset
boston= load_boston()
dfx = pd.DataFrame(boston.data, columns = boston.feature_names)
dfy = pd.DataFrame(boston.target, columns = ["MEDV"])
df = pd.concat([dfx,dfy],axis = 1)
df = pd.DataFrame(df)
type(df)
df.head()
df.shape
df.describe()

#Explore relationships between different columns
df.groupby(['AGE'])['MEDV'].mean()

#Create a new column
df['AGE_50'] = df['AGE'].apply(lambda x:x>50)
df['AGE_50'].value_counts()

grouby_twovar = df.groupby(['AGE_50','RAD','CHAS'])['MEDV'].mean()
grouby_twovar
grouby_twovar.unstack()
df['CHAS'].value_counts().plot(kind = 'bar')
df.corr(method = 'pearson')

import seaborn as sns
sns.heatmap(df.corr(), cmap = sns.cubehelix_palette(20, light = 0.95, dark = 0.15))

import matplotlib.pyplot as plt
plt.hist(df['MEDV'], bins = 50)
sns.distplot(df['MEDV'])
sns.jointplot(df['RM'], df['MEDV'], kind = 'scatter')
sns.kdeplot(df['RM'], df['MEDV'], shade = True)
sns.pairplot(df[['RM', 'AGE', 'LSTAT', 'DIS', 'MEDV']], kind="reg", plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.1}})

#Spit train test dataset

X_train, X_test, Y_train, Y_test = train_test_split(dfx, dfy, test_size = 0.33, random_state = 5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

#Run linear regression
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,Y_train)
Y_pred = lm.predict(X_test)

plt.scatter(Y_test,Y_pred)
plt.xlabel("prices")
plt.ylabel("Predicted prices ")
plt.title("Prices vs Predicted Prices")

#Calculate MSE
mse = sklearn.metrics.mean_squared_error(Y_test,Y_pred)
mse
#mse = 28.53, indicating its not such a good model

#Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
def create_polynomial_regression_model(degree):
    "Creates a polynomial Regression model for the given degree"
    poly_features = PolynomialFeatures(degree = degree)
    
    #transforms the existing features to higher degree features
    x_train_poly = poly_features.fit_transform(X_train)
    
    #fit the transformed features to linear regression
    poly_model = lm.fit(x_train_poly,Y_train)
    
    #predict on train set
    y_train_pred = poly_model.predict(x_train_poly)
    
    #predict on test set
    y_test_pred = poly_model.predict(poly_features.fit_transform(X_test))
    
    rmse_train = np.sqrt(mean_squared_error(Y_train,y_train_pred))
    r2_train = r2_score(Y_train, y_train_pred)
    
    rmse_test = np.sqrt(mean_squared_error(Y_test,y_test_pred))
    r2_test = r2_score(Y_test, y_test_pred)
    
    
    print("Model performance on training set")
    print("RMSE ",rmse_train)
    print("R2 Score ",r2_train)
    
    print("Model performance on test set")
    print("RMSE ",rmse_test)
    print("R2 Score ",r2_test)
    return;
    
create_polynomial_regression_model(2)
create_polynomial_regression_model(3)
create_polynomial_regression_model(4)
#best test performance is obtained with degree = 2