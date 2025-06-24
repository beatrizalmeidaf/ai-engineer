import nltk
from nltk.corpus import movie_reviews
from preprocessing import PreProcessing
from ngram_model import NGramModel
from naive_bayes import NaiveBayesClassifier

print("Carregando o dataset de reviews de filmes.")

neg_sents = movie_reviews.sents(categories='neg')
pos_sents = movie_reviews.sents(categories='pos')
all_sents_raw = neg_sents + pos_sents
labels = ['neg'] * len(neg_sents) + ['pos'] * len(pos_sents)
print(f"Dataset carregado com {len(all_sents_raw)} sentenças.")

processor = PreProcessing()

# teste pipeline de limpeza
all_sents_processed = [processor.process_pipeline(sent) for sent in all_sents_raw]

# teste ngrams
bigram_model = NGramModel(n=2)
bigram_model.fit(all_sents_processed)

print("\n5 bigramas mais comuns aprendidos pelo modelo:")
most_common = sorted(bigram_model.ngram_counts.items(), key=lambda item: item[1], reverse=True)
print(most_common[:5])


# teste prob condicionais
print("\nTestando Probabilidades Condicionais")

bigram_comum = ('year', 'old')
prob_comum = bigram_model.ngram_prob(bigram_comum)
print(f"Probabilidade de '{bigram_comum[1]}' após '{bigram_comum[0]}': {prob_comum:.4f}")

bigram_raro = ('the', 'pineapple')
prob_raro = bigram_model.ngram_prob(bigram_raro)
print(f"Probabilidade de '{bigram_raro[1]}' após '{bigram_raro[0]}': {prob_raro:.10f}")

# teste perplexidade
print("\nTestando a Perplexidade do Modelo ")

sentenca_normal = ['the', 'film', 'about', 'high', 'school', 'in', 'new', 'york', 'was', 'good']
ppl_normal = bigram_model.perplexity(sentenca_normal)
print(f"Sentença: '{' '.join(sentenca_normal)}'")
print(f"Perplexidade (confusão): {ppl_normal:.2f}")

print("-" * 20)

sentenca_ruim = ['the', 'pineapple', 'is', 'a', 'car']
ppl_ruim = bigram_model.perplexity(sentenca_ruim)
print(f"Sentença: '{' '.join(sentenca_ruim)}'")
print(f"Perplexidade (confusão): {ppl_ruim:.2f}")

# teste naive bayes
nb = NaiveBayesClassifier(n=2, laplace=1)
nb.fit(all_sents_processed, labels)

# testar classificação em uma sentença
test = ['a', 'marvelous', 'performance', 'by', 'an', 'amazing', 'cast']
pred = nb.predict(test)
print("Nova sentença:", test)
print("Classe prevista:", pred)