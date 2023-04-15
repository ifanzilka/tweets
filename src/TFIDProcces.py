import math
import pandas as pd


## Здесь мы смотрим как часто встречается слово во всем датасете и в конкретном примерие

class TFIDFProcessor:
    """
    term frequency - inverse document frequency
    """
    def __init__(self, rows: pd.DataFrame):
        self.rows = rows.copy()
        self.num_of_texts = rows.shape[0] ##Кол - во текстов обработанных
        self.num_of_apparitions = dict(TFIDFProcessor.binarize(rows).sum()) ##Словарь ({слово: кол - во повторений})
    
    @staticmethod##Если есть хоть один раз слово в тексте то 1, иначе 0
    def binarize(rows):
        return rows.apply(lambda r: [v & 1 for v in r])

    def compute_tf(self):
        ##Матрица где вместо чисел вес кажого слова (Например:"how unhappy some dogs like it though" -> у каждого слова вес 1/7 так как слов 7 у остальных 0)
        return self.rows.apply(lambda r: r / sum(r), axis = 1)
    
    def compute_idf(self):

        ## Возвращаю словарь где кажому слову кф встречания во всем тексе (10 логарифм (кол во встречаний / всего строк с текстом)  (среднее значение в тексте грубо говоря))
        term_importances = {}
        for w, c in self.num_of_apparitions.items():

            num_of_occs = 1.0 if c <= 0 else float(c)
            
            term_importances[w] = math.log10(float(self.num_of_texts) / num_of_occs)
        
        return term_importances
    
    def compute_tfidf(self):
        tf_dataset = self.compute_tf()

        for word, importance in self.compute_idf().items():
            ## word - слово
            ## importance -> cредний кф для слова
            
            tf_dataset[word] *= importance * 100000  ## Здесь матрицу текстов * умножаем на кф встречания во всем тексте.
        tf_dataset.fillna(0, inplace=True)
        return tf_dataset.astype(int)
    

if __name__ == "__main__":
    ## This main just tokenizator


    from Parser import ParserCsv
    from TextProccesing import TokenizerBase
    import pandas as pd
    from TextProcessor import PreProcessor
    from TextProcessor import binarize

    negative_tweets = ParserCsv.parse('./data/processedNegative.csv')
    positive_tweets = ParserCsv.parse('./data/processedPositive.csv')
    neutral_tweets = ParserCsv.parse('./data/processedNeutral.csv')

    

    neg_token = TokenizerBase(records=negative_tweets, binary_value = -1).process()
    pos_token = TokenizerBase(records=positive_tweets, binary_value = 1).process() 
    neu_token = TokenizerBase(records=neutral_tweets, binary_value = 0).process()

    all_df = neg_token + pos_token + neu_token
    df_token = pd.DataFrame(all_df)
    
    preprocessor = PreProcessor(df_token)
    preprocessor.execute()

    tfidf = TFIDFProcessor(preprocessor.df_x)
    tfidf.compute_tfidf()

    from collections import namedtuple
    Sentiment = namedtuple('Sentiment', [
    'type',
    'ordinal',
    ])

    Approach = namedtuple('Approach', [
        'binary',
        'counts',
        'tfidf',
        'df_y'
    ])

    Classifier = namedtuple('Classifier', [
        'model',
        'params'
    ])

    approaches = {
        'tokenization': None,
        'stemming': None,
        'lemmatization': None,
        's+m': None,
        'l+m': None,
    }

    sentiments = {
        'negative': Sentiment('negative', -1),
        'positive': Sentiment('positive', 1),
        'neutral': Sentiment('neutral', 0)
    }

    #approaches 
    approaches['tokenization'] = Approach(binarize(preprocessor.df_x), preprocessor.df_x, tfidf.compute_tfidf(), preprocessor.df_y)




    # def optimize_model_params(classifier: Classifier, x_train, y_train):
    #     gs = GridSearchCV(classifier.model(), param_grid=classifier.params, n_jobs=-1)
    #     gs.fit(x_train, y_train)
    #     return gs.best_params_, gs.best_score_

    # def find_best_model(df_x, df_y):
    #     X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.3)
    #     max_accuracy = 0
    #     best_model = None
    #     for name, model in models.items():
    #         print(f'optimizing {name}')
    #         best_params, best_accuracy = optimize_model_params(model, X_train, y_train)
    #         print(f'Best accuracy {best_accuracy} for model: {name}')
    #         if best_accuracy > max_accuracy:
    #             max_accuracy = best_accuracy
    #             best_model = Classifier(model.model, best_params)

    #     return best_model
    

    # from sklearn.linear_model import LogisticRegression
    # from sklearn.neighbors import KNeighborsClassifier
    # from sklearn.model_selection import train_test_split
    # from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
    # from sklearn.model_selection import GridSearchCV
    # from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
    # from sklearn.tree import DecisionTreeClassifier
    # from sklearn.svm import SVC


    # models = {
    # "logistic": Classifier(LogisticRegression, {"C": [1.0, 2.0, 0.5, 0.25], "solver": ('newton-cg', 'sag', 'saga'), "max_iter": [500]}),
    # "randomforest": Classifier(RandomForestClassifier, dict(n_estimators = [100, 300, 500], max_depth = [ 25, 30], min_samples_split = [2, 5], min_samples_leaf = [1, 2])),
    # "knn": Classifier(KNeighborsClassifier, dict(n_neighbors=range(2,7), algorithm=['ball_tree', 'kd_tree', 'auto'])),
    # "decisiontree": Classifier(DecisionTreeClassifier, dict(max_features=['sqrt', 'log2', None], criterion=["gini", "entropy"], min_samples_split=[2,3,4]))
    # }

    # trained_models = {}
    # for name, approach in approaches.items():
    #     print(f'Approach {name}')
    #     trained_models[name] = find_best_model(approach.counts, approach.df_y.sentiment)
    #     print()