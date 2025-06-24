import math
from collections import defaultdict

class NGramModel:
    def __init__(self, n, laplace=1):
        self.n = n # guarda o tamanho do n grama
        self.ngram_counts = defaultdict(int) # guardar contagens em um dicionario que começa em 0

        self.context_counts = defaultdict(int) # para um bigrama ('a', 'b'), o contexto é ('a',)
        self.laplace = laplace # fator de suavizacao
        self.vocab = set()

    def _generate_ngrams(self, tokens):
        # padding adiciona (n-1) tokens de início '<s>'
        padded_tokens = ['<s>'] * (self.n - 1) + tokens
        ngrams = [tuple(padded_tokens[i : i + self.n]) for i in range(len(padded_tokens) - self.n + 1)]
        return ngrams
    
    def fit(self, sentences) :
        self.vocab.add('<s>') # adiciona o token de inicio ao vocabulario

        for sentence in sentences:
            for word in sentence:
                self.vocab.add(word)

            ngrams = self._generate_ngrams(sentence)

            for ngram in ngrams:
                self.ngram_counts[ngram] +=1

                context = ngram[:-1]
                self.context_counts[context] +=1

        print("Treinamento Concluido")

    def ngram_prob(self, ngram):
        context = ngram[:-1]

        ngram_count = self.ngram_counts.get(ngram, 0)
        context_count = self.context_counts.get(context, 0)

        vocab_size = len(self.vocab)

        numerador = ngram_count + self.laplace
        denominador = context_count + self.laplace * vocab_size 

        if denominador == 0:
            return 0.0 # evitar divisao por zero se o contexto nunca foi visto
        
        prob = numerador / denominador

        return math.log2(prob)
    
    def sequencia_prob(self, sentence_tokens):
        total_log_prob = 0.0

        for ngram in self._generate_ngrams(sentence_tokens):
            total_log_prob += self.ngram_prob(ngram)

        return total_log_prob
    
    def perplexity(self, sentence_tokens):
        N = len(sentence_tokens)
        if N == 0:
            return float('inf') # perplexidade infinita para frase vazia

        log_prob = self.sequencia_prob(sentence_tokens)
        cross_entropy = -(1/N) * log_prob
        
        # calcula a perplexidade
        perplexity = math.pow(2, cross_entropy) # 2 elevado à entropia
        
        return perplexity