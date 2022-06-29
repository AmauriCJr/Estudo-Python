import numpy as np
import pandas as pd
import tensorflow as tf

tf.__version__

dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

#Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:,2] = le.fit_transform(x[:,2])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

print(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train) 
x_test = sc.transform(x_test)

#Building the ANN

#Inicializando
ann = tf.keras.models.Sequential()

#Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu')) #cria a primeira hidden layer

#Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))

#Adding the output layer
ann.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))
#caso não binário, a activation deve ser soft max

#Training the ANN

#Compilando
ann.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])#optimizer escolhido implementa o modelo estocastico de correção
#caso o resultado do modelo fosse não binário, o loss escolhido deve ser 'categorical'

#Treinando
ann.fit(x_train, y_train, batch_size = 32, epochs = 100) #batch_size é a quantidade de dados analisados antes de fazer a correção nos pesos

#Prevendo caso específico
print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)

y_pred = ann.predict(x_test)
y_pred = y_pred > 0.5 #Transforma a probabilidade em 0 ou 1
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
