import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score
import scikitplot.plotters as skplt

def top_features_by_class(tfidf_mat, features):

    for doc in tfidf_mat:
        doc

    return features


def top_features_by_doc(tfidf_mat, features):

    for doc in tfidf_mat:
        doc

    return features

def to_lower(gene_set):
    s = ' '.join(gene_set)
    s=s.lower()

    l = s.split(' ')

    return set(l)




def set_cl_probs(r, gene_class_probs):
    g = r['Gene']
    for i in range(1,9):
        if i in gene_class_probs[g]:
            r['cl'+str(i)] = gene_class_probs[g][i]

    return r


def remove_zero_rows(data):
    d = data.loc[(data[[1, 2, 3, 4, 5, 6, 7, 8]] > 0.01).any(axis=1) | (data[[1, 2, 3, 4, 5, 6, 7, 8]] < -0.01).any(axis=1)]

    return d


def evaluate_features(X, y, clf=None):
    """General helper function for evaluating effectiveness of passed features in ML model

    Prints out Log loss, accuracy, and confusion matrix with 3-fold stratified cross-validation

    Args:
        X (array-like): Features array. Shape (n_samples, n_features)

        y (array-like): Labels array. Shape (n_samples,)

        clf: Classifier to use. If None, default Log reg is use.
    """
    if clf is None:
        clf = LogisticRegression()

    probas = cross_val_predict(clf, X, y, cv=StratifiedKFold(random_state=8),
                               n_jobs=-1, method='predict_proba', verbose=2)
    pred_indices = np.argmax(probas, axis=1)
    classes = np.unique(y)
    preds = classes[pred_indices]
    print('Log loss: {}'.format(log_loss(y, probas)))
    print('Accuracy: {}'.format(accuracy_score(y, preds)))
    skplt.plot_confusion_matrix(y, preds)