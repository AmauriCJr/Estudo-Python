import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Encoding categorical data

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

#Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#Training the Multiple Linear Regression model on the Training set

from sklearn.linear_model import LinearRegression #devido a biblioteca escolhida não é necessário se preocupar com o melhor modelo de regressão ou com dummy variable, isso é feito automaticamente
regressor = LinearRegression() #Cria a ferramenta que realiza a regressão linear
regressor.fit(x_train, y_train) #treina a ferramenta de regressão nos dados de treino

#Predicting the Test set results

y_pred = regressor.predict(x_test)
np.set_printoptions(precision = 2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

#Making a single prediction (for example the profit of a startup with R&D Spend = 160000, Administration Spend = 130000, Marketing Spend = 300000 and State = 'California')

print(regressor.predict([[1, 0, 0, 160000, 130000, 300000]]))

#Getting the final linear regression equation with the values of the coefficients

print(regressor.coef_)
print(regressor.intercept_)
