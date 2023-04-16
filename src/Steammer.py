from nltk.stem import PorterStemmer

try:
    from src.TextProccesing import Extractor
    from src.TextProccesing import Processor
except:
    from TextProccesing import Extractor
    from TextProccesing import Processor
    

class Stemmer(Processor):
    
    def __init__(self,records = [], next_pipeline = None):
        """
        Args:
            @records -> list With comment
        """
        
        self.records = records
        self.stemmer = PorterStemmer() ##Stemmer (оставляет только основу слова)
        
        super().__init__(next_pipeline)
            
    def process(self):
        texts = []
        for text in self.records:
            words = []
            for w in Extractor.extract_words(text):
                words.append(self.stemmer.stem(w))
            texts.append(' '.join(words))
        return texts


    


if __name__ == "__main__":
    from Parser import ParserCsv
    from TextProccesing import TokenizerBase
    import pandas as pd
    from TextProccesing import PreProcessor
    from TextProccesing import binarize

    negative_tweets = ParserCsv.parse('./data/processedNegative.csv')
    positive_tweets = ParserCsv.parse('./data/processedPositive.csv')
    neutral_tweets = ParserCsv.parse('./data/processedNeutral.csv')

    

    neg_token = TokenizerBase(records=negative_tweets, binary_value = -1, next_pipeline=Stemmer(records = negative_tweets)).process_all()
    pos_token = TokenizerBase(records=positive_tweets, binary_value = 1, next_pipeline=Stemmer(records = positive_tweets)).process_all()
    neu_token = TokenizerBase(records=neutral_tweets, binary_value = 0, next_pipeline=Stemmer(records = neutral_tweets)).process_all()

    all_df = neg_token + pos_token + neu_token
    df_token = pd.DataFrame(all_df)


    print(df_token)
    preprocessor = PreProcessor(df_token)
    preprocessor.execute()