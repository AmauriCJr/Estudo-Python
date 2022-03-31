import matplotlib.pyplot as plt
import numpy as np

delimiter = ";"
ndelimiter = ","

f = open('C:/Users/Dell-Inspiron-14R/OneDrive/Área de Trabalho/Eletrônica/Dados - Data Analysis/Nomes.txt')

data = f.read()

f.close

fo = open('C:/Users/Dell-Inspiron-14R/OneDrive/Área de Trabalho/Eletrônica/Códigos Python/TESTE2.txt',"r+")
fo.truncate(0)
for car in data:
    if car == delimiter:
        car = ndelimiter
    fo.write(car)
fo.close


