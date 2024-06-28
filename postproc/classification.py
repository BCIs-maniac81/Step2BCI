# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 10:47:40 2020

@author: Hafed-eddine BENDIB
"""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import warnings

warnings.filterwarnings('ignore', 'Solver terminated early.*')


def knn_classifier(X_train, y_train, X_true, **kwargs):
    knn_ = KNeighborsClassifier(**kwargs)
    knn_.fit(X_train, y_train)
    y_pred = knn_.predict(X_true)
    return y_pred, knn_


def logreg_classifier(X_train, y_train, X_true, **kwargs):
    logreg_ = LogisticRegression(**kwargs)
    logreg_.fit(X_train, y_train)
    y_pred = logreg_.predict(X_true)
    return y_pred, logreg_


def gnb_classifier(X_train, y_train, X_true, **kwargs):
    gnb_ = GaussianNB(**kwargs)
    gnb_.fit(X_train, y_train)
    y_pred = gnb_.predict(X_true)
    return y_pred, gnb_


def mnb_classifier(X_train, y_train, X_true, *args, **kwargs):
    mnb_ = MultinomialNB(*args, **kwargs)
    mnb_.fit(X_train, y_train)
    y_pred = mnb_.predict(X_true)
    return y_pred, mnb_


def lda_classifier(X_train, y_train, X_true, shrinkage=None, *args, **kwargs):
    if isinstance(shrinkage, (int, float)):
        if shrinkage > 1:
            shrinkage = 1
        elif shrinkage <= 0:
            shrinkage = 0
    else:
        shrinkage = shrinkage

    lda_ = LinearDiscriminantAnalysis(shrinkage=shrinkage, *args, **kwargs)
    lda_.fit(X_train, y_train)
    y_pred = lda_.predict(X_true)
    return y_pred, lda_


def dt_classifier(X_train, y_train, X_true, *args, **kwargs):
    DecisionTree_ = DecisionTreeClassifier(*args, **kwargs)
    DecisionTree_.fit(X_train, y_train)
    y_pred = DecisionTree_.predict(X_true)
    return y_pred, DecisionTree_


def rf_classifier(X_train, y_train, X_true, **kwargs):
    RandomForest_ = RandomForestClassifier(**kwargs)
    RandomForest_.fit(X_train, y_train)
    y_pred = RandomForest_.predict(X_true)
    return y_pred, RandomForest_


def svm_classifier(X_train, y_train, X_true, fastmode=True,
                   gridsearch=False, cv=10, max_iter=50, **kwargs):
    hyper_params0 = [{'C': [0.1, 1, 10, 100], 'gamma': [0.01, 0.1, 1, 10]}]
    hyper_params1 = [{'C': [0.001, 0.01, 0.1, 1, 2, 5, 10, 20, 100, 1000], 'gamma': [0.01, 0.1, 1, 10, 'scale'],
                      'kernel': ["poly", "sigmoid", "rbf"], 'degree': [3, 4, 5]}]
    if fastmode and gridsearch:
        svm = GridSearchCV(SVC(max_iter=max_iter), param_grid=hyper_params0, cv=cv)
    elif fastmode and not gridsearch:
        svm = SVC(max_iter=max_iter, **kwargs)
    elif not fastmode and gridsearch:
        svm = GridSearchCV(SVC(max_iter=-1), param_grid=hyper_params1, cv=cv)
    else:
        svm = SVC(max_iter=-1, **kwargs)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_true)
    return y_pred, svm


#
# def svm_classifier(X_train, y_train, X_true, **kwargs):
#     hp0 = [{'C': [0.1, 1, 10, 100, 1000], 'gamma': [0.01, 0.1, 1, 10, 100]}]
#     hp1 = [{'C': [0.001, 0.01, 0.1, 1, 2, 5, 10, 20, 100, 1000], 'gamma': [0.01, 0.1, 1, 10, 'scale'],
#             'kernel': ["poly", "sigmoid", "rbf"], 'degree': [3, 4, 5]}]
#     if kwargs['default']:
#         params0 = dict(max_iter=-1, probability=False, cross_validation=True)
#     else:
#         params0 = dict(max_iter=100, probability=True, cross_validation=False)
#
#     if params0['cross_validation'] and kwargs['fastmode']:
#         hyper_params = hp0
#         svm_ = GridSearchCV(SVC(max_iter=params0['max_iter'], probability=params0['probability']),
#                             param_grid=hyper_params, cv=kwargs['cv'])
#
#     elif params0['cross_validation'] and not kwargs['fastmode']:
#         hyper_params = hp1
#         svm_ = GridSearchCV(SVC(max_iter=params0['max_iter'], probability=params0['probability']),
#                             param_grid=hyper_params, cv=kwargs['cv'], verbose=1)
#
#     elif not params0['cross_validation'] and not kwargs['fastmode']:
#         hyper_params = hp1
#         svm_ = GridSearchCV(SVC(max_iter=-1, probability=params0['probability']),
#                             param_grid=hyper_params, cv=kwargs['cv'])
#
#     elif not params0['cross_validation'] and kwargs['fastmode']:
#         hyper_params = {'C': 1.0, 'gamma': 'scale', 'kernel': 'rbf'}
#         svm_ = SVC(max_iter=-1, probability=False, **hyper_params)
#
#     svm_.fit(X_train, y_train)
#     y_pred = svm_.predict(X_true)
#     return y_pred, svm_


def train_lda(class1, class2):
    """
    Train the LDA algorithm
    :param class1: An array (observations x features) for class 1.
    :param class2: An array (observations x features) for class 2.
    :return: The projection matrix W
             The offset b
    """
    nclasses = 2
    nclass1 = class1.shape[0]
    nclass2 = class2.shape[0]

    # Class priors: in this case we have an equal number of training, so both priors = 0.5
    prior1 = nclass1 / float(nclass1 + nclass2)
    prior2 = nclass2 / float(nclass1 + nclass2)
    mean1 = np.mean(class1, axis=0)
    mean2 = np.mean(class2, axis=0)

    class1_centered = class1 - mean1
    class2_centered = class2 - mean2

    # Calculate the covariance between the features
    cov1 = class1_centered.T.dot(class1_centered) / (nclass1 - nclasses)
    cov2 = class2_centered.T.dot(class2_centered) / (nclass1 - nclasses)

    W = (mean2 - mean1).dot(np.linalg.pinv(prior1 * cov1 + prior2 * cov2))
    b = (prior1 * mean1 + prior2 * mean2).dot(W)
    return W, b


def apply_lda(test, W, b):
    """
    Applies a previously trained LDA to new data
    :param test: An array (features * trials) containing the data
    :param W: the project matrix W as calculated by train_lda()
    :param b: The offsets b as calculated by train_lda()
    :return:  A list containing a classlabel for each trial
    """
    ntrials = test.shape[1]
    prediction = []

    for i in range(ntrials):
        # result = W[0] * test(0, i] + W[1] * test[1, i] - b
        result = W.dot(test[:, i]) - b
        if result <= 0:
            prediction.append(1)
        else:
            prediction.append(2)
    return np.array(prediction)
