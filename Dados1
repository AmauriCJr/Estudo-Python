import matplotlib.pyplot as plt
import numpy as np

f = open('C:/Users/Dell-Inspiron-14R/OneDrive/Área de Trabalho/Eletrônica/Códigos Python/TESTE.txt', "r")

dados = f.read()

f.close

print(dados)

x = len(dados)

print(x)

i = 0

dados_t = np.arange((x//2) + 1)

for car in dados:
    if car != " ":
        dados_t[i] = car
    else:
        i = i + 1

print(dados_t)

dados_t = dados_t//dados_t

dados_t = str(dados_t)

y = len(dados_t)
print(y)

dados_t = dados_t[1:y-1]
print(dados_t)

print(type(dados_t))

fo = open('C:/Users/Dell-Inspiron-14R/OneDrive/Área de Trabalho/Eletrônica/Códigos Python/TESTE2.txt',"r+")
fo.truncate(0)
fo.write(dados_t)
fo.close
