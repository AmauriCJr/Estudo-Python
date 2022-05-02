#Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset

dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print(x)

print(y)

#Taking care of missing data

from sklearn.impute import SimpleImputer #biblioteca que possui ferramentas de preprocessamento de dados
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean') #escolher substituir valores faltantantes pela média
imputer.fit(x[:, 1:3]) #seleciona as colunas que vaõ ser analisadas
x[:, 1:3] = imputer.transform(x[:, 1:3]) #transforma as colunas de acordo com o que especificado

print(x)

#Encoding categorical data

#Encoding the Independent Variable

#irá tratar a primeira coluna que possui o nome dos países para que sejam organizadas de forma numérica sem ordem específica
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough') #cria a regra de transformação
x = np.array(ct.fit_transform(x)) #faz o fit e o transform ao mesmo tempo, diferente da função de transformação anterior

print(x)

#Encoding the Dependent Variable

#organiza as variáveis dependentes da mesma forma que a primeria coluna das independentes
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder() #cria a regra
y = le.fit_transform(y)

print(y)

#Splitting the dataset into the Training set and Test set

#divide os dados em duas partes, uma parte que servirá pra treinar a máquina e outra que servirá de teste
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1) #cria as variáveis usando os dados, destinando 20% do dados para realização dos testes

print(x_train)

print(x_test)

print(y_train)

print(y_test)

#Feature Scaling

#normaliza as variáveis para o processamento
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:]) #set faz o calculo da media e do desvio padrão, transform aplica o calculo da standarlization 
x_test[:, 3:] = sc.transform(x_test[:, 3:]) #aqui deve ser apenas aplicada a transformação, pq o set test foi retirado do resto e deve seguir a mesma regra do todo

print(x_train)

print(x_test)
