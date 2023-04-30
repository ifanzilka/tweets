import math
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

## Здесь мы смотрим как часто встречается слово во всем датасете и в конкретном примерие

class TFIDFProcessorPro:
    """
    term frequency - inverse document frequency
    """
    def __init__(self, df_x: list, dx_y:list):
        self.list_x = df_x ##list document
        self.list_y = dx_y #list category
    
        # Создание векторизатора
        self.vectorizer = TfidfVectorizer()



    def TFIDProcess(self):
        # Обучение векторизатора на корпусе документов
        self.vectorizer.fit(self.list_x)

        # Трансформация документов в векторы
        vectors = self.vectorizer.transform(self.list_x)
        lst = []
        for i, vector in enumerate(vectors):
            lst.append(vector.toarray()[0])
            #print(vector.toarray()[0].shape)
        #print(lst)
        self.vector = lst
        


    

if __name__ == "__main__":
    ## This main just tokenizator


    from Parser import ParserCsv
    from TextProccesing import TokenizerBase
    import pandas as pd
    from TextProccesing import PreProcessor
    from TextProccesing import binarize

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

    tfidf = TFIDFProcessorPro(list(preprocessor.df['tweet']), list(preprocessor.df_y['sentiment']))
    tfidf.TFIDProcess()
    #print("ok")

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

    # #approaches 
    # approaches['tokenization'] = Approach(binarize(preprocessor.df_x), preprocessor.df_x, tfidf.compute_tfidf(), preprocessor.df_y)


    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC


    def optimize_model_params(classifier: Classifier, x_train, y_train):
        gs = GridSearchCV(classifier.model(), param_grid=classifier.params, n_jobs=-1)
        gs.fit(x_train, y_train)
        return gs.best_params_, gs.best_score_

    def find_best_model(df_x, df_y):
        X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.3)
        max_accuracy = 0
        best_model = None
        for name, model in models.items():
            print(f'optimizing {name}')
            best_params, best_accuracy = optimize_model_params(model, X_train, y_train)
            print(f'Best accuracy {best_accuracy} for model: {name}')
            if best_accuracy > max_accuracy:
                max_accuracy = best_accuracy
                best_model = Classifier(model.model, best_params)

        return best_model
    



    models = {
    "logistic": Classifier(LogisticRegression, {"C": [1.0, 2.0, 0.5, 0.25], "solver": ('newton-cg', 'sag', 'saga'), "max_iter": [500]}),
    "randomforest": Classifier(RandomForestClassifier, dict(n_estimators = [100, 300, 500], max_depth = [ 25, 30], min_samples_split = [2, 5], min_samples_leaf = [1, 2])),
    "knn": Classifier(KNeighborsClassifier, dict(n_neighbors=range(2,7), algorithm=['ball_tree', 'kd_tree', 'auto'])),
    "decisiontree": Classifier(DecisionTreeClassifier, dict(max_features=['sqrt', 'log2', None], criterion=["gini", "entropy"], min_samples_split=[2,3,4]))
    }

    result = find_best_model(tfidf.vector, tfidf.list_y)
    # optimizing logistic
    # Best accuracy 0.878620052279107 for model: logistic
    # optimizing randomforest
    # Best accuracy 0.8604320268154575 for model: randomforest
    # optimizing knn
    # Best accuracy 0.7290293880309813 for model: knn
    # optimizing decisiontree
    # Best accuracy 0.8593181645757323 for model: decisiontree

