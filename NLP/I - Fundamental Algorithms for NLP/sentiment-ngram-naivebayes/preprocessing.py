import re
import nltk 

from nltk.corpus import movie_reviews, stopwords
from nltk.tokenize import word_tokenize

try:
    stopwords.words('english')
    movie_reviews.words()
except LookupError:
    nltk.download('stopwords')
    nltk.download('movie_reviews')
    nltk.download('punkt')


class PreProcessing():
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text) # sub => remove pontuações e numeros
        text = re.sub(r'<.*?>', '', text) # remove tags html
        text = re.sub(r'\s+', ' ', text).strip()  # remove espaços extras
        return text
    
    def tokenizer(self, text):
        tokens = word_tokenize(text)
        return tokens
    
    def remove_stopwords(self, tokens):
        tokens = [x for x in tokens if x not in self.stop_words]
        return tokens
    
    def process_pipeline(self, raw_text):
        """ Executa o fluxo de pre processamento """
        raw_text = ' '.join(raw_text)
        cleaned_text = self.clean_text(raw_text)

        tokens = self.tokenizer(cleaned_text)

        filtered_tokens = self.remove_stopwords(tokens)

        return filtered_tokens