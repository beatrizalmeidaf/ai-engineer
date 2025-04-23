entradas = [1, 2, 3, 4, 5]

valores_esperados = [23, 26, 29, 32, 35]

# parâmetros iniciais (valores que serão ajustados)

peso = 0.1  # multiplicador das entradas (w)
viés = 0  # valor fixo que será somado
taxa_de_aprendizado = 0.01  # o quanto os ajustes são feitos em cada passo
epocas = 6000  # quantas vezes o processo será repetido

def prever(entrada):
    return peso * entrada + viés

for epoca in range(epocas):
    # calcula as saídas para todas as entradas usando a função prever
    predicoes = [prever(entrada) for entrada in entradas]

    # calcula os erros: diferença entre os valores esperados e as predições
    erros = [esperado - predito for predito,esperado in zip(predicoes, valores_esperados)]

    # determina o sinal de cada erro (positivo ou negativo)

    sinais = [1 if erro > 0 else -1 for erro in erros]

    # média dos valores absolutos dos erros
    custo_medio = sum(abs(erro) for erro in erros) / len(erros)

    print(f"Época: {epoca}, Viés: {viés:.2f}, Peso: {peso:.2f}, Custo: {custo_medio:.2f}")

    # ajusta o peso (peso += ajuste baseado nos sinais e entradas)
    peso += taxa_de_aprendizado * sum(sinal * entrada for sinal, entrada in zip(sinais, entradas))

    # ajusta o viés (viés += ajuste baseado nos sinais)
    viés += taxa_de_aprendizado * sum(sinais) / len(entradas)


# testa o modelo com novos valores
novas_entradas = [6, 7]  # novas entradas para prever
novos_valores_esperados = [38, 41]  # valores esperados para essas entradas

# calcula as predições para as novas entradas
novas_predicoes = [prever(entrada) for entrada in novas_entradas]

# mostra as entradas, os valores esperados e as predições
for entrada, esperado, predito in zip(novas_entradas, novos_valores_esperados, novas_predicoes):
    print(f"Entrada: {entrada}, Esperado: {esperado}, Previsto: {predito:.2f}")