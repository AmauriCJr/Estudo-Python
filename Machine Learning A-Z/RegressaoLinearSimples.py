#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset

dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

print(x_train)
print(x_test)
print(y_train)
print(x_test)

#Training the Simple Linear Regression model on the Training set

from sklearn.linear_model import LinearRegression #linarregression é a classe
regressor = LinearRegression() #regressor é um objeto da classe acima
regressor.fit(x_train, y_train) #encaixa as variaveis de treino no objeto regressor e treina o modelo

#Predicting the Test set results

y_pred = regressor.predict(x_test)

#Visualising the Training set results

plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salário X Experiência (Training set)')
plt.xlabel('Anos de Experiência')
plt.ylabel('Salário')
plt.show()

#Visualising the Test set results

plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salário X Experiência (Test set)')
plt.xlabel('Anos de Experiência')
plt.ylabel('Salário')
plt.show()

#Making a single prediction (for example the salary of an employee with 12 years of experience)

print(regressor.predict([[12]])) #prevê o valor do salário para 12 anos de experiência

#Getting the final linear regression equation with the values of the coefficients

print(regressor.coef_) #coeficiente angular
print(regressor.intercept_) #constante
