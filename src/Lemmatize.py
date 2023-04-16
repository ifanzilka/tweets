
from nltk.stem import  WordNetLemmatizer
from nltk.corpus import wordnet
import nltk

try:
    from src.TextProccesing import Extractor
    from src.TextProccesing import Processor
except:
    from TextProccesing import Extractor
    from TextProccesing import Processor


nltk.download('wordnet')

class Lemmatizer(Processor):
    def __init__(self, records=[], next_pileline = None ):
        self.records = records
        self.lemmatizer = WordNetLemmatizer() ##Возвращает исходное слово если найдено в базе
        super().__init__(next_pileline)

        
    def process(self):
        texts = []
        for text in self.records:
            words = []
            for w in Extractor.extract_words(text):
                words.append(self.lemmatizer.lemmatize(w))#, pos=Lemmatizer.get_wordnet_pos(w)))
            texts.append(' '.join(words))
        return texts

    @classmethod
    def get_wordnet_pos(cls, word):
        pos = nltk.pos_tag([word])
        treebank_tag = pos[0][1]
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN
        

if __name__ == "__main__":
    from Parser import ParserCsv
    from TextProccesing import TokenizerBase
    from TextProccesing import PreProcessor
    import pandas as pd

    negative_tweets = ParserCsv.parse('./data/processedNegative.csv')
    positive_tweets = ParserCsv.parse('./data/processedPositive.csv')
    neutral_tweets = ParserCsv.parse('./data/processedNeutral.csv')

    neg_token_lem = TokenizerBase(records=negative_tweets, binary_value = -1, next_pipeline=Lemmatizer(records = negative_tweets)).process_all()
    pos_token_lem = TokenizerBase(records=positive_tweets, binary_value = 1, next_pipeline=Lemmatizer(records = positive_tweets)).process_all()
    neu_token_lem = TokenizerBase(records=neutral_tweets, binary_value = 0, next_pipeline=Lemmatizer(records = neutral_tweets)).process_all()

    all_lemmed = pd.DataFrame(neg_token_lem + pos_token_lem + neu_token_lem)
    pp_stemmed = PreProcessor(all_lemmed)
    pp_stemmed.execute()

