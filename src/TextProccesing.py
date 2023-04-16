
import pandas as pd
import re
from collections import Counter

class Extractor:
    """
    Возвращает из тескта чисто слова (массив слов)
    """
    ascii_word_regex = re.compile(r"[0-9A-Za-z]+")
    
    @classmethod
    def extract_words(self, text):
        return self.ascii_word_regex.findall(text)


class Processor:
    """

    Main class for Text Proccesing
    
    """
    
    def __init__(self, instance):
        
        if instance is None:
            self._instance = instance
        elif not isinstance(instance, Processor):
            raise TypeError(f"incorrect processor usage {instance}")
        else:
            self._instance = instance
            self.records = instance.records
        
    def process_all(self):

        """
        Start all class process
        
        """
        if self._instance is None:
            
            return self.process()
        else:
            ## Start sub class processing
            self.records = self._instance.process_all()
            return self.process()    
    
    def process(self):
        raise NotImplementedError("incorrect processor usage")

class TokenizerBase(Processor):
    
    def __init__(self, records, binary_value, next_pipeline = None):
        """
        args:
            entiment -> name dataframe(negative, neural, positive)
            records -> list with text

        """
        self.records = records
        self.binary_value = binary_value
        self.next_pipeline = next_pipeline
    
        #self.sentiment = sentiment#sentiments.get(sentiment) #Example: Sentiment(type='negative', ordinal=-1)
        super().__init__(next_pipeline)

    def count_tokens(self, text):
        """
            Разбивает текст на слова и их количество возращает словарь
        """
        words = Extractor.extract_words(text)
        wc = Counter(words)
        wc['tweet'] = text
        return dict(wc)

    def format_to_row(self, wc):
        wc['sentiment'] = self.binary_value
        return wc

    def process(self):
        """
            return dict {text, binary and count word}
        """
       
        wc = (self.count_tokens(text) for text in self.records) ## Генератор которые возвращает словарь
        final_rows = [self.format_to_row(wcount) for wcount in wc] ## Ставит бинарное значение каждомо словарю (-1, 0, 1) (negative, neutral, positive)
        return final_rows
    


class PreProcessor:
    def __init__(self, df):
        """
        Create Data For Machine Lerning
        """
        
        self.df = df ## pandas dataframe
    
    def rearrange(self):
        ##Удаляю текст оригинальный и банарное значение
        df = self.df
        cols = list(df.columns)
        cols.remove('sentiment')
        cols.remove('tweet')
        self.df = df[['sentiment', 'tweet'] + cols]
        
    def split(self):
        # df -> original 
        # df_x -> matrix with 0  / 1
        # df_y -> binary res and text original #['sentiment', 'tweet']
        ## nan заменяю на - 0
        df = self.df
        df.fillna(0, inplace=True) 
        self.df_x = df.iloc[:, 2:].astype(int)
        self.df_y = df.iloc[:, :2]
        
    def clean(self):

        ##Убирает дубликаты и числовой текст
        df = self.df
        self.df.drop_duplicates(subset='tweet', inplace=True, keep='last')
        cols = filter(lambda c: not c.isnumeric(), df.columns)
        self.df = df[cols]
        
    def execute(self):
        self.clean()
        self.rearrange()
        self.split()


## Return 1 if word in text else 0
def binarize(processed_rows):
    return processed_rows.apply(lambda r: [v & 1 for v in r])


if __name__ == "__main__":

    #from src.Parser import ParserCsv
    from Parser import ParserCsv

    negative_tweets = ParserCsv.parse('./data/processedNegative.csv')
    positive_tweets = ParserCsv.parse('./data/processedPositive.csv')
    neutral_tweets = ParserCsv.parse('./data/processedNeutral.csv')




    neg_token = TokenizerBase(records=negative_tweets, binary_value = -1).process_all()
    pos_token = TokenizerBase(records=positive_tweets, binary_value = 1).process_all()
    neu_token = TokenizerBase(records=neutral_tweets, binary_value = 0).process_all()

    print(neg_token[0])
    all_df = neg_token + pos_token + neu_token

    print(len(all_df))
    #print(all_df)
    
    

    df_token = pd.DataFrame(all_df)
    #print(df_token)

    preprocessor = PreProcessor(df_token)
    preprocessor.execute()
    print(preprocessor.df_y)