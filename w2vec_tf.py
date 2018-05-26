import numpy as np
import tensorflow as tf
import pandas as pd
from doc2vec import *

# corpus_raw = 'He is the king . The king is royal . She is the royal  queen '
# # convert to lower case
# corpus_raw = corpus_raw.lower()
#
#
# words = []
# for word in corpus_raw.split():
#     if word != '.': # because we don't want to treat . as a word
#         words.append(word)
# words = set(words) # so that all duplicate words are removed
# word2int = {}
# int2word = {}
# vocab_size = len(words) # gives the total number of unique words
# for i,word in enumerate(words):
#     word2int[word] = i
#     int2word[i] = word
#
# # raw sentences is a list of sentences.
# raw_sentences = corpus_raw.split('.')
# sentences = []
# for sentence in raw_sentences:
#     sentences.append(sentence.split())
#
#
#

text = pd.read_csv('training_text', sep="\|\|", index_col=[0], engine='python')
var = pd.read_csv('training_variants', index_col = [0])


data = pd.concat([var, text], axis=1)
corpus = data.Text.tolist()
sent_iter = corpus2sentences_generator(data.Text.tolist())
a=0
for i in sent_iter:
    a+=1

b=1