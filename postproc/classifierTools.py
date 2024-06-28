# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 14:08:32 2020

@author: Hafed-eddine BENDIB
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, matthews_corrcoef
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import postproc.classification as classify
import postproc.classificationPlotter as plotter
import warnings

warnings.filterwarnings('ignore', 'Solver terminated early.*')


def train_test_clf(features=None, targets=None, test_size=0.2, random_state=0, **kwargs):
    split_features_targets = train_test_split(features, targets,
                                              test_size=test_size, random_state=random_state)
    return split_features_targets


def average_score_clf(clf_inst=None, features=None, targets=None, test_size=0.2, random_state=0, **kwargs):
    if clf_inst is None:
        print("Instance invalide ")
    else:
        avg_score = cross_val_score(estimator=clf_inst, X=features, y=targets, **kwargs).mean()
    return avg_score


def report_clf(estimator=None, y_true=None, y_pred=None, conf_matrix_plot=False, **kwargs):
    report = classification_report(y_true, y_pred, labels=None)
    cm = confusion_matrix(y_true, y_pred, labels=estimator.classes_, **kwargs)
    if conf_matrix_plot:
        disp = ConfusionMatrixDisplay(cm, display_labels=estimator.classes_)
        disp.plot()
        plt.show()
    return report, estimator.classes_


def accuracy_clf(y_true=None, y_pred=None):
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy


def correct_rate_clf(y_true=None, y_pred=None):
    correct_rate = (len(np.where(y_true - y_pred == 0)[0])) * (100 / len(y_true))
    return correct_rate


def best_k_knn_scores(features, targets, order_n=10, ts=.4, rs=4, plot=True, title='', xlabel='', ylabel=''):
    split_features_targets = train_test_clf(features, targets, test_size=ts, random_state=rs)
    scores = []
    x_ax = np.arange(1, order_n + 1, 1)
    X_train = split_features_targets[0]
    X_true = split_features_targets[1]
    y_train = split_features_targets[2]
    y_true = split_features_targets[3]
    for k in x_ax:
        y_pred, _ = classify.knn_classifier(X_train, y_train, X_true, n_neighbors=k)
        scores.append(accuracy_score(y_true, y_pred))
    scores = np.array(scores)
    best_k = int((np.where(scores == scores.max())[0]).max()) + 1
    if plot:
        plotter.knn_k_values_plot(scores, x_ax, title, xlabel, ylabel)
    return scores, best_k


def cv_parameter_tuning(features, targets, order_n=10, cv=10, score_type="accuracy", plot=True,
                        title='', xlabel='', ylabel=''):
    k_scores = []
    x_ax = np.arange(1, order_n + 1, 1)
    for k in x_ax:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(estimator=knn, X=features, y=targets, cv=cv, scoring=score_type)
        k_scores.append(scores.mean())
    k_scores = np.array(k_scores)
    best_k = int((np.where(k_scores == k_scores.max())[0]).max()) + 1
    if plot:
        plotter.cv_parameter_plot(k_scores, x_ax, title, xlabel, ylabel)

    return k_scores, best_k


def estimator_select(features=None, targets=None, cv=10, ts=0.2, rs=0, splitted_data=None,
                     inputs=None, with_predict=False):
    if splitted_data is not None:
        splitted_data = splitted_data
    else:
        if features is not None and targets is not None:
            splitted_data = train_test_clf(features, targets, test_size=ts, random_state=rs)
        else:
            print('échec de la classification')
            return 'No data'

    model_params = {"KNN": {"estimator": KNeighborsClassifier(),
                            "parameters": {"n_neighbors": range(2, 21)}
                            },
                    "logistic_regression": {"estimator": LogisticRegression(solver="liblinear"),
                                            "parameters": {"C": [0.1, 1, 2, 5, 10]}
                                            },
                    "GaussianNB": {"estimator": GaussianNB(),
                                   "parameters": {"var_smoothing": np.logspace(0, -9, num=100)},
                                   },
                    # "MultinomialNB" : {"estimator" : MultinomialNB(),
                    #                    "parameters" : {"alpha" : [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5,
                    #                                    1e-4, 1e-3, 1e-2, 1e-1, 1.] }
                    #                    },
                    "LDA": {"estimator": LinearDiscriminantAnalysis(),
                            "parameters": {"shrinkage": [None],
                                           "solver": ["svd", "lsqr", "eigen"]}
                            },
                    "rfc": {"estimator": RandomForestClassifier(),
                            "parameters": {"max_depth": [None], "n_estimators": [100, 200, 300],
                                           "criterion": ["gini", "entropy"], "min_samples_split": [2, 3, 4]}
                            },
                    "svm": {"estimator": SVC(),
                            "parameters": {"C": [0.001, 0.01, 0.1, 1, 2, 5, 10, 20, 100, 1000],
                                           "kernel": ["rbf", "linear", "poly", "sigmoid"],
                                           "gamma": ["auto", "scale"]}
                            },
                    "decision_tree": {"estimator": DecisionTreeClassifier(),
                                      "parameters": {"max_depth": [None, 10, 100, 1000, 10000, 100000],
                                                     "splitter": ["best", "random"]}
                                      }
                    }
    gscv_s = []
    for model_name, estim_params in model_params.items():
        GSCV = GridSearchCV(estimator=estim_params["estimator"], param_grid=estim_params["parameters"], cv=cv)
        GSCV.fit(splitted_data[0], splitted_data[2])
        if with_predict and inputs is not None:
            if np.asarray(inputs).ndim < 2:
                inputs = np.asarray(inputs).reshape(1, -1)
                y_pred = GSCV.predict(inputs)
            else:
                y_pred = GSCV.predict(inputs)
        else:
            y_pred = None
        gscv_s.append({"estimator": model_name,
                       "best_estimator": GSCV.best_estimator_,
                       "best parameters": GSCV.best_params_,
                       "mean cross_validation accuracy": GSCV.best_score_,
                       "test accuracy": GSCV.score(splitted_data[1], splitted_data[3]),
                       })

    return gscv_s, y_pred
