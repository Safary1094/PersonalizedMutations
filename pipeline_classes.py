import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class preprocessing(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, data):

        # data = data.dropna()
        data = data.reset_index()

        return data

class ItemSelector(BaseEstimator, TransformerMixin):

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        return data[self.key]


class gene_class_prob(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.gene_class_probs = {}

    def fit(self, X, *_):

        gene_set = set(X['Gene'].tolist())

        for gene_name in gene_set:
            g = X.loc[X.Gene == gene_name]
            probs = g.Class.value_counts(normalize=True, sort=False)

            self.gene_class_probs[gene_name] = probs

        return self

    def transform(self, X, *_):
        class_label = ['cl1', 'cl2', 'cl3', 'cl4', 'cl5', 'cl6', 'cl7', 'cl8', 'cl9']

        t = np.zeros((len(X), len(class_label)))

        for ind, r in X.iterrows():
            gene_name = X.loc[ind, 'Gene']
            if gene_name not in self.gene_class_probs:
                for i in range(1, 9):
                    t[ind, i] = 0
            else:
                for i in range(1, 9):
                    if i in self.gene_class_probs[gene_name]:
                        t[ind, i] = self.gene_class_probs[gene_name][i]

        return t


class cat2binary(BaseEstimator, TransformerMixin):

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        return data[self.key]