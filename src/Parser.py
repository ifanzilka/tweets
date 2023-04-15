import pandas as pd
import re

class ParserCsv:
    non_word_regex = re.compile(r"[^0-9^A-Z^a-z^ ]") ##Замена все что не входит в этот список на пустоту (удаляю знаки припинания)
    
    @classmethod
    def filter_non_words(cls, text):
        return ParserCsv.non_word_regex.sub('', text).lower() ## в нижние регистр
    
    @staticmethod
    def parse(path):
        """
        Args:
            @path -> path in csv file
        Return:
            List with comments (lower case)
        """
        df = pd.read_csv(path)
        df = pd.DataFrame(data={'col': df.items()}, index = range(df.shape[1]))
        
        ## Применяю функцию к каждой строке
        return list(df['col'].apply(lambda r: ParserCsv.filter_non_words(r[0])))
    

if __name__ == "__main__":
    negative_tweets = ParserCsv.parse('./data/processedNegative.csv')
    positive_tweets = ParserCsv.parse('./data/processedPositive.csv')
    neutral_tweets = ParserCsv.parse('./data/processedNeutral.csv')

    print(negative_tweets)
