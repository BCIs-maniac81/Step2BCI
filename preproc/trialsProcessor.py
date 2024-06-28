# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 10:31:17 2019

Author: Hafed-eddine BENDIB
"""

import numpy as np
import postproc.classifierTools as tools


# def trials_extract(inputData=None, trials=None, targets=None, trial_size=8, window=[0.5, 2.5], n_channels=None,
#                    offset=0, version="graz", sfreq=256):
#     """
#     trial_size: exprimé en secondes.
#     """
#     if n_channels is None:
#         n_channels = inputData.shape[-1]
#     # combien de classes exist-il ?
#     cls_list, cls_count = targets_count(targets)
#     cls_targets = list([])
#     cls_trials = list([])
#     trials_data = {}
#     for (i, j) in enumerate(cls_list):
#         cls_targets.append(np.where(np.asarray(targets) == j)[0])
#         cls_trials.append(np.asarray(trials[cls_targets[i]]))
#         if version.lower() == "graz":
#             trials_data.update({j: np.zeros((n_channels, len(cls_targets[i]), sfreq * (trial_size + offset)))})
#         elif version.lower() == "berlin":
#             win = np.arange(int(window[0] * sfreq), int(window[1] * sfreq))
#             trials_data.update({j: np.zeros((len(cls_targets[i]), int(sfreq * (window[1] - window[0] + offset)),
#                                              n_channels))})
#     for (k, class_) in enumerate(trials_data.keys()):
#         for ch in range(n_channels):
#             for (l, trial) in enumerate(cls_trials[k]):
#                 if version.lower() == "graz":
#                     trials_data[class_][ch, l, :] = inputData[trial + (sfreq * offset): trial + (
#                             trial_size * sfreq), ch]
#                     # trials_data[class_][ch, l, :] = inputData[trial - (sfreq * trial_size//2) + (sfreq * offset):
#                     #                                           trial + (trial_size//2 * sfreq), ch]
#
#                 elif version.lower() == "berlin":
#                     trials_data[class_][l, :, ch] = inputData[trial - (sfreq * offset) + win[0]: trial + (win[-1] + 1),
#                                                     ch]
#     return trials_data


def targets_count(targets_):
    cls_list = []
    cls_count = 0
    for cls_ in targets_:
        if cls_ not in cls_list:
            cls_list.append(cls_)
            cls_count += 1
    cls_list = sorted(cls_list, key=None, reverse=False)
    return cls_list, cls_count


def trials_extract(inputData: object = None, trials: object = None, targets: object = None, trial_size: object = 8, window: object = [0.5, 2.5],
                   n_channels: object = None,
                   offset: object = 0, version: object = "graz", sfreq: object = 256) -> object:
    """
    trial_size: exprimé en secondes.
    """
    if n_channels is None:
        n_channels = inputData.shape[-1]
    # combien de classes exist-il ?
    cls_list, cls_count = targets_count(targets)
    cls_targets = list([])
    cls_trials = list([])
    trials_data = {}
    for (i, j) in enumerate(cls_list):
        cls_targets.append(np.where(np.asarray(targets) == j)[0])
        cls_trials.append(np.asarray(trials[cls_targets[i]]))
        if version.lower() == "graz":
            trials_data.update({j: np.zeros((len(cls_targets[i]), sfreq * (trial_size + offset), n_channels))})
        elif version.lower() == "berlin":
            win = np.arange(int(window[0] * sfreq), int(window[1] * sfreq))
            trials_data.update({j: np.zeros((len(cls_targets[i]), int(sfreq * (window[1] - window[0] + offset)),
                                             n_channels))})
    for (k, class_) in enumerate(trials_data.keys()):
        for ch in range(n_channels):
            for (l, trial) in enumerate(cls_trials[k]):
                if version.lower() == "graz":
                    trials_data[class_][l, :, ch] = inputData[trial + (sfreq * offset): trial + (
                            trial_size * sfreq), ch]
                    # trials_data[class_][ch, l, :] = inputData[trial - (sfreq * trial_size//2) + (sfreq * offset):
                    #                                           trial + (trial_size//2 * sfreq), ch]

                elif version.lower() == "berlin":
                    trials_data[class_][l, :, ch] = inputData[trial - (sfreq * offset) + win[0]: trial + (win[-1] + 1),
                                                    ch]
    return trials_data


def data_class_composer(data_cls1, data_cls2, classes=None):
    if classes is None:
        classes = ['class1', 'class2']
    return {classes[0]: data_cls1, classes[1]: data_cls2}


def train_true_composer(splitted_data=None, classes=None, target_codes=None):
    if classes is None:
        classes = ['class1', 'class2']
    if target_codes is None:
        target_codes = [0, 1]
    train, true, y_train, y_true = splitted_data
    train_ = {classes[0]: train[y_train == target_codes[0]], classes[1]: train[y_train == target_codes[1]]}
    true_ = {classes[0]: true[y_true == target_codes[0]], classes[1]: true[y_true == target_codes[1]]}
    return train_, true_


def split_data_classes(inputData, classes=None, for_classification=False, target_codes=[-1, 1], ts=0.25, rs=123):
    if classes is None:
        cl1 = list(inputData.keys())[0]
        cl2 = list(inputData.keys())[1]
    else:
        cl1 = classes[0]
        cl2 = classes[1]
    features = np.concatenate((inputData[cl1], inputData[cl2]), axis=0)
    targets = np.concatenate((np.full((inputData[cl1].shape[0],), target_codes[0]),
                              np.full((inputData[cl2].shape[0],), target_codes[1])),
                             axis=0)
    if for_classification:
        return features, targets
    X_train, X_true, y_train, y_true = tools.train_test_clf(features=features, targets=targets,
                                                            test_size=ts, random_state=rs)
    X_train_cl1 = X_train[y_train == target_codes[0], :, :]
    X_train_cl2 = X_train[y_train == target_codes[1], :, :]
    X_true_cl1 = X_true[y_true == target_codes[0], :, :]
    X_true_cl2 = X_true[y_true == target_codes[1], :, :]
    return data_class_composer(X_train_cl1, X_train_cl2, classes=classes), \
           data_class_composer(X_true_cl1, X_true_cl2, classes=classes)




if __name__ == "__main__":
    print(f"This is the test section of trials processing module")
