# Classificador de Sentimentos com Análise Probabilística de Texto

**Projeto de Processamento de Linguagem Natural (PLN)** com foco em modelagem de linguagem baseada em n-gramas e classificação de sentimentos usando o classificador Naive Bayes.

## Objetivo

Desenvolver um pipeline completo de PLN que:

* Pré-processa textos brutos 
* Modela sequências de texto com n-gramas (unigrama, bigrama, trigrama).
* Treina um classificador Naive Bayes para prever sentimentos (positivo ou negativo).

## Etapas do Pipeline

### 1. Pré-processamento (`preprocessing.py`)

* Limpeza com expressões regulares
* Tokenização e normalização (minúsculas, remoção de pontuação)

### 2. Modelagem de Linguagem com N-gramas (`ngram_model.py`)

* Construção de unigramas, bigramas e trigramas
* Cálculo de probabilidades, entropia cruzada e perplexidade
* Análise de escassez de dados e aplicação de suavização de Laplace

### 3. Classificação com Naive Bayes (`naive_bayes.py`)

* Vetorização utilizando n-gramas
* Estimativa das probabilidades P(palavra|classe) e P(classe)
* Aplicação do classificador com e sem suavização
* Testes com textos novos

## Dataset Sugerido

* **IMDb Movie Reviews (versão reduzida)**

  * Tarefa: Classificação binária de sentimentos
  * Fonte: `nltk.corpus.movie_reviews`

