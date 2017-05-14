import os, sys
from  tensorflow.contrib.learn import DNNClassifier
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)
from helpers import features_helpers

class DNN(BaseEstimator, ClassifierMixin):

    def __init__(self, n_classes, type="w2v", hidden_units=[10, 20, 10], num_features=100, context=10, method=1):
        #if type=="w2v":
            #self.model = w2v_helpers.get_word2vec(num_features, context, method)
        self.type = type
        self.classifier = DNNClassifier(hidden_units=hidden_units, n_classes=n_classes)

    def pre_transformX(self, df, colnames, df_test=None, n_gram=None):
        data = None
        if self.type =="w2v":
            data = features_helpers.create_sentences(df, colnames)
            data = features_helpers.transform_to_w2v_sentences(data, self.model)
            return data.as_matrix()
        else:
            x_train, x_test = features_helpers.transform_to_bow(df, df_test, colnames, n_gram)
            return x_train, x_test

    def pre_transformY(self, df, list_dict):
        y = map(lambda w: list_dict.index(w), list(df))
        return np.array(y)

    def fit(self, X, y=None):
        self.classifier.fit(x=X, y=y, steps=200)

    def predict(self, X, y=None):
        return self.classifier.predict(X)

    def evaluate(self, X, Y):
        return self.classifier.evaluate(x=X,y=Y)["accuracy"]

    def score(self, X, y, sample_weight=None):
        return super(DNN, self).score(X, y, sample_weight)
