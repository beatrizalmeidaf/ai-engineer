from collections import defaultdict
import math
from ngram_model import NGramModel

class NaiveBayesClassifier:
    def __init__(self, n, laplace=1):
        self.n = n
        self.laplace = laplace
        self.models = {} # guarda um modelo de n-grama para cada classe (positivo, negativo)
        self.priors = {} # guarda a probabilidade de cada classe (a priori)

    def fit(self, sentences, labels):

        print("Iniciando o treinamento do Classificador...")
        
        # separar sentenças e contar documentos por classe
        sents_by_class = defaultdict(list)
        doc_count_by_class = defaultdict(int)
        for sent, label in zip(sentences, labels):
            sents_by_class[label].append(sent)
            doc_count_by_class[label] += 1
            
        # calcular as probabilidades a priori P(Classe)
        total_docs = len(labels)
        for label, count in doc_count_by_class.items():
            self.priors[label] = math.log(count / total_docs)
            
        # treinar um NGramModel para cada classe
        for label, sents in sents_by_class.items():
            print(f"Treinando o especialista para a classe: '{label}'")
            # cria o especialista
            model = NGramModel(n=self.n, laplace=self.laplace)
            # treina o especialista SÓ com os dados da sua classe
            model.fit(sents)
            # guarda o especialista treinado
            self.models[label] = model
            
        print("Treinamento do classificador concluído!")

    def predict(self, sentence_tokens):
        scores = {}
        # calcula um score para cada classe
        for label, model in self.models.items():
            # começa com o prior
            prior = self.priors[label]
            # soma a likelihood 
            likelihood = model.sequencia_prob(sentence_tokens)
            
            scores[label] = prior + likelihood
            
        # retorna a classe que teve o maior score
        return max(scores, key=scores.get)