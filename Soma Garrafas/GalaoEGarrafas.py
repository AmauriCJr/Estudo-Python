
def prep(garrafas, galao):
    garrafas = sorted(list(garrafas)) #Organiza o vetor em ordem crescente
    # Declara variáveis que serão usadas durante a operação
    resultado = [] 
    temp = []
    soma = []
    soma_maior = [0]
    soma_maior2 = [0]
    comprimento = len(garrafas)
    menor_posi = 0
    #fim das declarações
    soma_valores(resultado, garrafas, galao, temp, 0, 0, soma, soma_maior) #Chama a função que realiza todas as somas dentro do vetor
    
    soma_maior = sorted(soma_maior) #Organiza o vetor contendo os maiores valores de soma em ordem crescente
   

    if len(resultado) == 0: #Checa se o vetor resultado está vazio, o que indica que a soma total do galão não foi atingida
        soma_valores(resultado, garrafas, soma_maior[len(soma_maior) - 1], temp, 0, 0, soma, soma_maior2) # chama a função novamente, porém como valor de soma desejado o valor da maior soma ao invés do valor do galão
        

    for i in range(0, len(resultado)): #loop que verifica todos os resultados e escolhe aquele que utiliza o menor numero de garrafas
        if len(resultado[i]) < comprimento:
            comprimento = len(resultado[i])
            menor_posi = i


    return resultado[menor_posi], soma_maior[len(soma_maior) - 1]
    


    

def soma_valores(resultado, garrafas, galao, temp, endereco, temp_soma, soma, soma_maior):
    
    
    if (temp_soma - galao == 0): # verifica se o galão está cheio, verificando se o volume restante é igual a zero
        resultado.append(list(temp)) #manda os valores armazenados na variavel temp pra variavel resultado e finaliza o loop
        return

    for i in range(endereco, len(garrafas)): #lê o vetor de "endereco" ate o comprimente de garrafas
        if (temp_soma + garrafas[i]) <= galao: #verifica se a soma dos valores é menor ou igual ao valor do galão
            temp.append(garrafas[i]) # se for menor ele adiciona o valor no vetor de garrafas usadas pra encher o galão
            soma = temp_soma + garrafas[i]
            for j in range(0, len(soma_maior)): # faz um loop pra guardar os valores das somas, para verificar qual é a maior caso não atinja o valor do galao
                if soma > soma_maior[j]:
                    soma_maior.append(soma)
            soma_valores(resultado, garrafas, galao, temp, i + 1, temp_soma + garrafas[i], soma, soma_maior) #chama a função novamente pra repetir até encher o galão
            # Enviar o i + 1 na linha acima foi a sacada pra n serem lidos numeros repetidos
            temp.remove(garrafas[i]) # remove o valor ja somado da lista que estourou a soma para testar outro
            


garrafas = [1.2, 2.4, 4.3, 3.2] # valor em litros de cada garrafa

galao = 10 # valor em litros do galao



resultado, soma = prep(garrafas, galao) # chama a função que realiza as operações



print("Resposta: ", resultado, ", sobra ", round(galao - soma, 2), "L" ) #imprime os resultados
