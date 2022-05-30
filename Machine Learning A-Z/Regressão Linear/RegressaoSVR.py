import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

print(x)

print(y)

y = y.reshape((len(y), 1)) #transforma em um vetor 2D com uma coluna e várias linhas

print(y)

#Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

print(x)

print(y)

#Training the SVR model on the whole dataset

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x, y)

#Predicting a new result

sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])).reshape(-1, 1)) #função reshape aplicada pois estava ocorrendo um erro e essa foi a sugestão dada.
#função inverse transform remove o Feature Scaling, é feito o predict do valor 6.5, mas esse valor tem que passar pelo Feature Scaling para devolver o valor correto

#Visualising the SVR results

plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_x.inverse_transform(x), sc_y.inverse_transform(regressor.predict(x).reshape(-1, 1)), color = 'blue')
plt.title('Salário X Cargo (SVR)')
plt.xlabel('Cargo')
plt.ylabel('Salário')
plt.show()

Visualising the SVR results (for higher resolution and smoother curve)

x_grid = np.arange(min(sc_x.inverse_transform(x)), max(sc_x.inverse_transform(x)), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color = 'red')
plt.plot(x_grid, sc_y.inverse_transform(regressor.predict(sc_x.transform(x_grid)).reshape(-1, 1)), color = 'blue')
plt.title('Salário X Cargo (SVR)')
plt.xlabel('Cargo')
plt.ylabel('Salário')
plt.show()
#Esse modelo não lida muito bem com pontos fora da curva
