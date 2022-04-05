import matplotlib.pyplot as plt
import numpy as np

def checa_maior(garrafas):

    maior = 0
    for i in garrafas:
        if i > maior:
            maior = i
    return maior

def soma_valores(garrafas,i,j):
    soma = garrafas[i] + garrafas[j]
    return soma

def soma_pares(garrafas):
    i = 0
    j = 0
    soma_total = 0
    soma_array = [0] 
    tam = len(garrafas)
    while i < tam:
        while j < tam:
            if i != j:
                soma = soma_valores(garrafas, i, j)
                soma_total = soma_total + soma
                if soma_total <= galao:
                    soma_array.append(soma_total)
                soma_total = 0
            j = j + 1
        i = i + 1
        comp = j - i
        j = j - comp
    soma_array.pop(0)
    return soma_array



numero = 3

garrafas = [1.00 , 4.50, 1.00]

galao = 5

quantidade = len(garrafas)

somas = garrafas
somas.append(0)
print ("somas = ", somas)
valor = 0
contagem = 1
while contagem < quantidade:
    teste = soma_pares(somas)
    valor = checa_maior(teste)
    print("teste = ", teste)
    somas = np.concatenate((somas, teste), axis=None)
    print ("somas = ", somas)
    print("valor = ", valor)
    contagem = contagem + 1
    print("contagem = ", contagem)

print("sucesso valor = ", valor)








