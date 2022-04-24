def verifica_repeticao(garrafas, res):
    #if len(garrafas) == 1:
    #    garrafas.append(0)
    vetor =[]
    res = sorted(list(res))
    garrafas = sorted(list(garrafas))
    if len(garrafas) == 0:
        return 0
    for i in range(0, len(garrafas)):
        for j in range(0, len(res)):
            if garrafas[i] == res[j]:
                vetor.append(garrafas[i])
                res[j] = -1
    if len(garrafas) != len(vetor):
        return 1
    else:
        return 0   
