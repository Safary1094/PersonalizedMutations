import  pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import linear_model, model_selection, grid_search
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np
from pipeline_classes import *
from aux_fun import *
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import confusion_matrix
from doc2vec import *

text = pd.read_csv('training_text', sep="\|\|", index_col=[0], engine='python')
var = pd.read_csv('training_variants', index_col = [0])


data = pd.concat([var, text], axis=1)
#
data=data.dropna()
data = data.reset_index()


data = pd.concat([data, pd.get_dummies(data.Gene)], axis=1)

# make pipeline

tfidf =  TfidfVectorizer(min_df=2, use_idf=True, stop_words = 'english', max_features=1000, ngram_range=(2,3))
log_reg = linear_model.LogisticRegression(penalty='l1', C=0.15)
svd = TruncatedSVD(n_components=50)

w2vec = get_word2vec(corpus2sentences_generator, 'w2v_model')

w2vec_vectorizer = Vectorizer(w2vec)

text_column = 'Text'
categ_column = list(set(data['Gene'].tolist()))

feature_processing_pipeline = FeatureUnion(transformer_list=[
                                    # binary
                                    # ('class_processing', ItemSelector(key=categ_column)),

                                    # text
                                    ('text_processing', Pipeline(steps=[
                                        ('selector', ItemSelector(key=text_column)),
                                        # ('tfidf', tfidf),
                                        ('w2vec', w2vec_vectorizer),
                                        # ('svd', svd)
                                    ]))
                                ])

estimator = Pipeline(steps=[

    ('feature_processing', feature_processing_pipeline),
    ('model_fitting', log_reg)
])

parameters_grid = {
    # 'feature_processing__text_processing__tfidf__max_features' : [500],
    # 'feature_processing__text_processing__tfidf__max_df' : [6, 10],
    # 'feature_processing__text_processing__tfidf__min_df' : [2, 4],
    'model_fitting__C' : [0.15]
}

grid_cv = model_selection.GridSearchCV(estimator, parameters_grid, n_jobs=4, scoring='accuracy', cv=8)

grid_cv.fit(data, data.Class.values)

# print('Best score ' + str(grid_cv.best_score_))
# print(grid_cv.best_params_)

#
# feature_processing_pipeline.fit(data, data.Class.values)
# transformed_data = feature_processing_pipeline.transform(data)
# transformed_data = transformed_data.todense()
#
# tsvd = TruncatedSVD(n_components=2)
# data_pca = tsvd.fit_transform(transformed_data)
# # plt.figure
# # plt.scatter(x=data_pca[:,0], y=data_pca[:,1], c=data.Class.tolist(), label= data.Class.tolist())
#
#
# estimator.fit(data, data.Class.values)
#
# f=pd.DataFrame(estimator._final_estimator.coef_).T
# fea = tfidf.get_feature_names()
#
# f['f_name']=fea
#
# f2 = remove_zero_rows(f)

# tfidf_res = tfidf.fit_transform(data.Text)


# for s in sent_iter:
#     print(s)


# mean_embedded = w2vec_vectorizer.fit_transform(data.Text)

evaluate_features(data, data.Class, estimator)

1
