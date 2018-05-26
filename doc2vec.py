import gensim
import nltk
import os
import numpy as np
import string
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
import time
from sys import getsizeof

class corpus2sentences_generator(object):
    # corpus -> sentences -> [word_1, ... word_n]   - generator()
    def __init__(self, *arrays):
        self.arrays = arrays

    def __iter__(self):
        stop = stopwords.words('english') + list(string.punctuation)
        for array in self.arrays:
            for document in array:
                for sent in nltk.sent_tokenize(document):
                    yield nltk.word_tokenize(sent)


def get_word2vec(sentences, location):

    if os.path.exists(location):
        print('Found {}'.format(location))
        model = gensim.models.Word2Vec.load(location)
        return model

    print('{} not found. training model'.format(location))
    model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
    print('Model done training. Saving to disk')
    model.save(location)
    return model


class Vectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.wv.syn0[0])

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        stop = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', ' - ', '.', '/', ':', ';', '<', '=', '>',
                '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']

        X_transformed=[]

        for document in X:
            document = document.lower()
            for s in stop:
                document = document.replace(s, '')

            document = document.replace('  ', ' ')
            document = document.replace('   ', ' ')
            tokenized_doc = document.split(' ')

            t = self.word2vec.wv[tokenized_doc]

            X_transformed.append(np.mean(t, axis=0))

            print([len(X_transformed)])
        return np.array(X_transformed)