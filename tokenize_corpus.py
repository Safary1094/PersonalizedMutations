import string
from nltk.corpus import stopwords
import nltk
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer


text = pd.read_csv('training_text', sep="\|\|", index_col=[0], engine='python')
var = pd.read_csv('training_variants', index_col = [0])

data = pd.concat([var, text], axis=1)

data=data.dropna()
data = data.reset_index()

corpus = data.Text.tolist()

# stop = stopwords.words('english') + list(string.punctuation)
#
# WINDOW_SIZE = 2
#
# for document in corpus:
#     for sent in nltk.sent_tokenize(document):
#         for word_index, word in enumerate(sent):
#             for nb_word in sent[max(word_index - WINDOW_SIZE, 0): min(word_index + WINDOW_SIZE, len(sent)) + 1]:
#                 if nb_word != word:
#                     data.append([word, nb_word])

vectorizer = CountVectorizer()
vectorizer.fit([corpus[0], corpus[1]])
1