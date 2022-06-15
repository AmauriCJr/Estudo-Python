!pip install apyori

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Data Preprocessing

dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
  transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])#le cada valor do dataset e guarda ele em uma outra variável no formato lista de listas string

#Training the Apriori model on the dataset 

from apyori import apriori
rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2) #o transactions antes da igualdade é o nome do parâmetro da função
#min_suport = 3 vezes ao dia * 7 dias por semana/ 7501 transações totais
#min_confidence - o valor padrão era 0.8, mas isso siginifica 80% de valores coincidentes, o que não devolvia resultado nenhum, então foi escolhido 0.2
#min_lift - o valor 3 foi sugerido por ser um valor bom segundo a experiência do professor
#length - como o desejo é relacionar dois produtos, os valores são ajustados para devolver a relação entre dois produtos  

#Displaying the first results coming directly from the output of the apriori function

results = list(rules)
results

#Putting the results well organised into a Pandas DataFrame

def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

#Displaying the results non sorted

resultsinDataFrame

#Displaying the results sorted by descending lifts

resultsinDataFrame.nlargest(n = 10, columns = 'Lift')
