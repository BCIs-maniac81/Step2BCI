# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 10:31:17 2019

Author: Hafed-eddine BENDIB
"""

import numpy as np
from pandas import read_table
import csv
import mat4py
from scipy.io import loadmat
import datetime
import os


def read_text_file(path='path', method='r', fformat="openBCI", ch_ax=-1, t_ax=0):
    file = open(path, method)
    list_ = []
    while True:
        line = file.readline()
        if line.startswith("%") or line.startswith("#"):
            continue
        if len(line) == 0:
            break
        if fformat.lower() == 'openbci':
            line = line.split(",")
        else:
            line = line.split()
        list_.append([])
        for j in range(0, len(line)):
            list_[ch_ax].append(float(line[j]))
    arr = np.array(list_)
    file.close()
    return arr


def read_csv_file(path=None, method='r', header="WITH", ch_ax=0, t_ax=-1):
    file = open(path, method, newline='')
    read_object = csv.reader(file)
    list_ = []

    for row in read_object:
        if row[0].startswith("%") or row[0].startswith("#"):
            continue
        list_.append(row)
    if header == "WITH":
        arr = np.array(list_[1:], dtype=float)
    else:
        arr = np.array(list_, dtype=float)
    file.close()
    return arr


def read_dat_file(path=None):
    data = np.array(read_table(path)).copy()
    return data


def read_matlab_file(full_path=None, verbosity=False, dtype="EEG"):
    if not full_path:
        path_dir = str(input("Please enter a valid path to Dataset:"))  # directory
        file_name = str(input("Please enter a valid name for Dataset:"))  # without extension id
        full_path = os.path.join(path_dir, "{}.mat".format(file_name))
    raw_mat_file = loadmat(full_path, struct_as_record=True)
    data_attr_ = list(raw_mat_file.keys())
    if data_attr_ is not None and verbosity:
        print("dataset attributes are: {}".format(data_attr_))
    X = []
    targets = []
    trials = []
    pt = 0
    expr_ = 0
    trials = raw_mat_file["mrk"][0][0][0][-1]
    targets = raw_mat_file["mrk"][0][0][1][-1]
    X = raw_mat_file["cnt"]
    n_samples_, n_channels_ = raw_mat_file["cnt"].shape
    attributes = {"sfreq": raw_mat_file["nfo"][0][0][0][0][0],
                  "Classes": [str_[0] for str_ in raw_mat_file["nfo"]["classes"][0][0][0]],
                  "Channels": [str_[0] for str_ in (raw_mat_file["nfo"][0][0][2][0])]}
    if verbosity:
        print("Number of samples: ", n_samples_)
        print("Number of channels: ", n_channels_)
        print("Dataset Attributes: ", attributes)
    return X, trials, targets, attributes


def read_graz_file(full_path=None, verbosity=False, dtype="EEG"):
    if not full_path:
        path_dir = str(input("Please enter a valid path to Dataset:"))  # directory
        file_name = str(input("Please enter a valid name for Dataset:"))  # without extension id
        full_path = os.path.join(path_dir, "{}.mat".format(file_name))
    raw_data = mat4py.loadmat(full_path)["data"]

    data_attr_ = list(raw_data[0].keys())
    if data_attr_ is not None and verbosity:
        print("dataset attributes are: {}".format(data_attr_))
    X = []
    targets = []
    trials = []
    pt = 0
    expr_ = 0
    while pt < len(raw_data):
        data = raw_data[pt]["X"]
        target = raw_data[pt]["y"]
        if 'artifacts' in raw_data[pt].keys():
            artifact = raw_data[pt]["artifacts"]
            artifact_idx = np.where(np.array(artifact) == 1)[0]
            target = list(np.delete(target, artifact_idx))
        X.append(data)
        targets.append(target)

        for i in range(pt):
            expr_ = expr_ + len(X[i])
        if isinstance(raw_data[pt]["trial"][0], list):
            trials_list = [x for xs in raw_data[pt]["trial"] for x in xs]
        else:
            trials_list = raw_data[pt]["trial"]
        trial = [expr_ + k for k in trials_list]

        # trial = [expr_ + k for k in (sum(raw_data[pt]["trial"], []))]
        if 'artifacts' in raw_data[pt].keys():
            trial = list(np.delete(trial, artifact_idx))
        trials.append(trial)
        pt += 1
        expr_ = 0

    attributes = {"sfreq": raw_data[0]["fs"], "Classes": raw_data[0]["classes"]}

    if 'gender' in raw_data[0].keys():
        attributes.update({"Gender": raw_data[0]["gender"]})

    if 'age' in raw_data[0].keys():
        attributes.update({"Age": raw_data[0]["age"]})
    X = np.array(sum(X, []))
    trials = np.array(sum(trials, []))
    targets = np.array(sum(targets, []))
    return X, trials, targets, attributes


def read_competition3_dataset1(directory_path, verbosity=True):
    """
     Lire le jeu de données 1 de la compétition 3 à partir d'un répertoire.

     Paramètres :
         directory_path (str) : chemin d'accès au répertoire contenant les fichiers de l'ensemble de données.

     Retour : Un tuple de tableaux numpy contenant les données d'apprentissage, les données de test et les véritables
     étiquettes.
    """
    sfreq = 1000
    # Load train data
    train_data = loadmat(directory_path + '/Competition_train.mat')
    train_data['sfreq'] = sfreq
    train_data['X'] = (train_data['X'].astype('double')).transpose(0, -1, -2)

    # Map label -1 --- to --- 0
    train_data['Y'][train_data['Y'] == -1] = 0
    # Load test data
    test_data = loadmat(directory_path + '/Competition_test.mat')
    test_data['sfreq'] = sfreq
    test_data['X'] = (test_data['X'].astype('double')).transpose(0, -1, -2)
    # Load true labels
    true_labels = read_text_file(directory_path + '/true_labels.txt').astype('int')
    test_data['Y'] = true_labels
    # Map label -1 --- to --- 0
    test_data['Y'][test_data['Y'] == -1] = 0

    if verbosity:
        print('Nombre des essais (l\'ensemble train) : ', train_data['X'].shape[0])
        print('Nombre des essais (l\'ensemble test) : ', test_data['X'].shape[0])
        print('Nombre des canaux: ', train_data['X'].shape[-1])
        print('Nombre d\'échantillons pour chaque essai (train): ', train_data['X'].shape[1])
        print('Nombre d\'échantillons pour chaque essai (test): ', test_data['X'].shape[1])

    return train_data, test_data


def write_csv_file(inputData=None, path=None, method='w'):
    if np.ndim(inputData) == 1:
        inputData = inputData.reshape(len(inputData), 1)
    pos = 10
    date_ = datetime.now()
    if path is None:
        path = 'file_saved_' + (str(date_)[:pos] + '_' + str(date_)[pos + 1:pos + 2] +
                                '-' + str(date_)[pos + 4:pos + 6] + '-' + str(date_)[pos + 7:pos + 9]) + '.csv'
    file = open(path, method, newline='')
    write_object = csv.writer(file)
    write_object.writerows(inputData)
    file.close()


if __name__ == '__main__':
    print(f"This is the test section of IOfiles module")



