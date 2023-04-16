from symspellpy import SymSpell, Verbosity
import pkg_resources

try:
    from src.TextProccesing import Processor
    from src.TextProccesing import Extractor
except:
    from TextProccesing import Processor
    from TextProccesing import Extractor


class MisspellingsCorrector(Processor):

    ## Здесь мы пробуем преобразовать слова близкие к данному (исправляются орфографические ошибки)
    
    def init_symspell(self):
        sym_spell = SymSpell(max_dictionary_edit_distance=1)
        dictionary_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_dictionary_en_82_765.txt")
        sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
        self.sym_spell = sym_spell
        
    def __init__(self, records=[], next_pipeline = None):
        self.records = records
        self.init_symspell()
        super().__init__(next_pipeline)
    

    
    def correct_word(self, word: str) -> str:
        corrections = self.sym_spell.lookup(word, Verbosity.CLOSEST, ignore_token=r"\w+\d")
        if len(corrections) == 0:
            return ''
        else:
            return corrections[0].term
        
    def process(self):
        texts = []
        for text in self.records:
            words = []
            for w in Extractor.extract_words(text):
                words.append(self.correct_word(w))
            texts.append(' '.join(words))
        return texts
    
if __name__ == "__main__":
    
    from Parser import ParserCsv
    from TextProccesing import TokenizerBase
    from TextProccesing import PreProcessor
    import pandas as pd

    from Steammer import Stemmer

    negative_tweets = ParserCsv.parse('./data/processedNegative.csv')
    positive_tweets = ParserCsv.parse('./data/processedPositive.csv')
    neutral_tweets = ParserCsv.parse('./data/processedNeutral.csv')

    neg_token_corr = TokenizerBase(records=negative_tweets, binary_value = -1, next_pipeline=Stemmer(records = negative_tweets, next_pipeline=MisspellingsCorrector(records=negative_tweets))).process_all()
    pos_token_corr = TokenizerBase(records=positive_tweets, binary_value = 1, next_pipeline=Stemmer(records = positive_tweets, next_pipeline=MisspellingsCorrector(records=positive_tweets))).process_all()
    neu_token_corr = TokenizerBase(records=neutral_tweets, binary_value = 0, next_pipeline=Stemmer(records = neutral_tweets, next_pipeline=MisspellingsCorrector(records=neutral_tweets))).process_all()

    all_stemmed_corrected = pd.DataFrame(neg_token_corr + pos_token_corr + neu_token_corr)
    pp_stemmed_corrected = PreProcessor(all_stemmed_corrected)
    pp_stemmed_corrected.execute()