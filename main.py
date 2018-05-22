import  pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import linear_model, model_selection, grid_search
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
import numpy as np
from pipeline_classes import *

text = pd.read_csv('training_text', sep="\|\|", index_col=[0], engine='python')
var = pd.read_csv('training_variants', index_col = [0])


data = pd.concat([var, text], axis=1)
#
data=data.dropna()
data = data.reset_index()

# make pipeline

tfidf =  TfidfVectorizer(max_features=200, min_df=5, use_idf=True, stop_words = 'english')
gene_class = gene_class_prob()
log_reg = linear_model.LogisticRegression(penalty='l1')


text_column = 'Text'
# text_indices = np.array([(column in text_columns) for column in data.columns], dtype = bool)

categ_column = ['Class', 'Gene']
# categ_indices = np.array([(column in categ_columns) for column in data.columns], dtype = bool)

estimator = Pipeline(steps=[

    ('preprocessing', preprocessing()),

    ('feature_processing', FeatureUnion(transformer_list=[
        # class to probabilities
        ('class_processing', Pipeline(steps=[
            ('selector', ItemSelector(key=categ_column)),
            ('class_to_probs', gene_class)
        ])),

        # text
        ('text_processing', Pipeline(steps=[
            ('selector', ItemSelector(key=text_column)),
            ('hot_encoding', tfidf)
        ])),


    ])),
    ('model_fitting', log_reg)
]
)

parameters_grid = {
    'feature_processing__text_processing__hot_encoding__max_features' : [500],
    'model_fitting__C' : [0.25, 0.15]
}

grid_cv = model_selection.GridSearchCV(estimator, parameters_grid, n_jobs=-1,scoring = 'accuracy', cv=10)

grid_cv.fit(data, data.Class.values)

# estimator.fit(data, data.Class.values)


# tfidf = tfidf.fit(data['Text'].tolist())
# tfidf_mat = tfidf.transform(data['Text'].tolist())


# d = tfidf_mat.todense()

# gene_set = set(data.Gene.tolist())
# gene_class_probs={}
# for gene_name in gene_set:
#     gene_class_probs[gene_name]=calculate_class_probs(data, gene_name)
#
# x = pd.DataFrame(d)
#
#
#
#
#
#
# c = data[class_label].as_matrix()
# x = np.concatenate((x,c), axis=1)
# y = data.Class.values
#
# log_reg = linear_model.LogisticRegression()
#
#
#
# log_reg_score = model_selection.cross_val_score(log_reg, x, y, n_jobs=10, cv=10)



1
