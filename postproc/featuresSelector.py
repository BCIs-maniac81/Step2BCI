# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 11:43:05 2020

@author: Hafed-eddine BENDIB
"""
import numpy as np
from sklearn.decomposition import PCA
import matplotlib
from sklearn import neighbors, linear_model, naive_bayes, discriminant_analysis, ensemble, svm, tree
from sklearn.model_selection import train_test_split, GridSearchCV
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.pipeline import Pipeline
import postproc.featuresPlotter as plotter

matplotlib.use('TkAgg')


def SFS_select(inputData=None, targetData=None, validation_size=0.25, forward=True, floating=False, k_feat_=4,
               rs=123456789, verbosity=True, all_cpu=True, cv=5, fitting_all_data=True, refit=True, clfs=None):
    if clfs is None:
        clfs = ["knn", "logreg", "gaussnb", "multinb", "lda", "rfc", "svm", "dtc"]
    if isinstance(k_feat_, tuple):
        if not (max(k_feat_) <= inputData.shape[-1]):
            k_feat = list(((inputData.shape[-1] // 2) + 1,))
        else:
            k_feat = list(k_feat_)
    elif isinstance(k_feat_, int) or isinstance(k_feat_, np.int32) or isinstance(k_feat_, np.int64) \
            or isinstance(k_feat_, np.int128):
        if not (k_feat_ <= inputData.shape[-1]):
            k_feat = list(((inputData.shape[-1] // 2) + 1,))
        else:
            k_feat = list((k_feat_,))
    else:
        return 'k_feat_ doit être un entier ou liste d\'entiers'

    estimators = []
    seq_feat_select = []
    pipeline_ = []
    gscv = []
    if not fitting_all_data:
        train, true, train_targets, true_targets = train_test_split(inputData, targetData,
                                                                    test_size=validation_size, random_state=rs)
    else:
        train = inputData
        train_targets = targetData

    classifiers = {"knn": neighbors.KNeighborsClassifier(),
                   "logreg": linear_model.LogisticRegression(solver="liblinear"),
                   "gaussnb": naive_bayes.GaussianNB(),
                   "multinb": naive_bayes.MultinomialNB(),
                   "lda": discriminant_analysis.LinearDiscriminantAnalysis(),
                   "rfc": ensemble.RandomForestClassifier(),
                   "svm": svm.SVC(),
                   "dtc": tree.DecisionTreeClassifier()
                   }
    if all_cpu:
        n_jobs = -1
    else:
        n_jobs = 1

    if verbosity:
        v = 2
    else:
        v = 0
    hyper_params = [[{'sfs__estimator__n_neighbors': [1, 5, 10, 15, 20], 'sfs__k_features': k_feat}],
                    [{'sfs__estimator__C': [0.1, 1, 2, 5, 10], 'sfs__k_features': k_feat}],
                    [{'sfs__estimator__var_smoothing': list(np.logspace(0, -9, num=100)), 'sfs__k_features': k_feat}],
                    [{'sfs__estimator__alpha': [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.],
                      'sfs__k_features': k_feat}],
                    [{'sfs__estimator__shrinkage': [None, "auto"], 'sfs__estimator__solver': ["svd", "lsqr", "eigen"],
                      'sfs__k_features': k_feat}],
                    [{'sfs__estimator__max_depth': [None], 'sfs__estimator__n_estimators': [100, 200, 300],
                      'sfs__estimator__criterion': ["gini", "entropy"], 'sfs__estimator__min_samples_split': [2, 3, 4],
                      'sfs__k_features': k_feat}],
                    [{'sfs__estimator__C': [0.1, 1, 2, 5, 10, 20, 100],
                      'sfs__estimator__kernel': ["rbf", "linear", "poly", "sigmoid"], 'sfs__k_features': k_feat}],
                    [{'sfs__estimator__max_depth': [None, 10, 100, 1000, 10000, 100000],
                      'sfs__estimator__splitter': ["best", "random"], 'sfs__k_features': k_feat}]]

    for idx, key in enumerate(classifiers.keys()):
        if clfs.__contains__(key):
            estimators.append(classifiers[key])
            seq_feat_select.append(
                SFS(estimator=estimators[idx], k_features=k_feat, forward=forward, floating=floating, verbose=v,
                    n_jobs=n_jobs, cv=cv))
            pipeline_.append(Pipeline(steps=[("sfs", seq_feat_select[idx]), (key, estimators[idx])]))
            gscv.append(
                GridSearchCV(estimator=pipeline_[idx], param_grid=hyper_params[idx], scoring="accuracy", n_jobs=n_jobs,
                             cv=cv, refit=refit))
        else:
            estimators.append(None)
            seq_feat_select.append(None)
            pipeline_.append(None)
            gscv.append(None)
    print(len(gscv))

    for idx in range(len(gscv)):
        if gscv[idx] is not None:
            gscv[idx].fit(train, train_targets)

    # best score_ : gscv[idx].best_score_
    # best params_ : gscv[idx].best_params_
    # gscv[0].best_estimator_.steps[0][1].__dir__()
    # gscv[0].best_estimator_.steps[0][1].k_score_
    # gscv[0].best_estimator_.steps[0][1].subsets_)
    # gscv[0].best_estimator_.steps[0][1].k_feature_idx_
    return gscv, pipeline_


def PCA_select(inputData=None, targetData=None, n_comp=3, plot=False, pca_plot=False, ts=0.3, rs=123, **kwargs):
    pca = PCA(n_components=n_comp)
    selected_data = pca.fit_transform(inputData)
    if plot:
        cmap = matplotlib.colors.ListedColormap(["red", "green"])
        if pca_plot:
            pca_data = PCA(3).fit_transform(inputData)
            split_data = train_test_split(pca_data, targetData, test_size=ts, random_state=rs)
            title = "Principal Components Plot"
            labels = ["PC1", "PC2", "PC3"]
        else:
            split_data = train_test_split(inputData, targetData, test_size=ts, random_state=rs)
            title = "First 3 Features plot"
            labels = ["Feature 1", "Feature 2", "Feature 3"]
        fig = None
        axis = []
        plotter.pca_selector_plot(split_data, fig=fig, axis=axis, title=title, labels=labels, cmap=cmap, **kwargs)
        matplotlib.pyplot.tight_layout()
        matplotlib.pyplot.show()
    else:
        split_data = None
    return selected_data, split_data


def estimate_pca_dimensions(train, true, threshold=0.95, plot=True):
    features = np.concatenate((train, true), axis=0)
    pca = PCA().fit(features)
    explained_var_ratio = pca.explained_variance_ratio_
    cumulative_var_ratio = np.cumsum(explained_var_ratio)
    n_components = np.argmax(cumulative_var_ratio >= threshold) + 1
    # plot elbow curve
    if plot:
        matplotlib.pyplot.figure()
        matplotlib.pyplot.plot(cumulative_var_ratio)
        matplotlib.pyplot.xlabel('Nombre de composantes PCA')
        matplotlib.pyplot.ylabel('Taux de variance expliquée cumulée (max = 1.0)')
        matplotlib.pyplot.axvline(x=n_components, color='r', linestyle='--')
        # Adding text annotation
        matplotlib.pyplot.text(n_components + 1, threshold, str(threshold * 100) + ' %', color='r')
        matplotlib.pyplot.xticks(list(matplotlib.pyplot.xticks()[0]) + [n_components])
        matplotlib.pyplot.grid()
        matplotlib.pyplot.show()

    return n_components


if __name__ == '__main__':
    print('This is a test section for features selector module')
