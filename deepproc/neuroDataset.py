import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data.dataloader import Dataset
from torchvision import transforms


class LoadSplitDataset(object):
    def __init__(self, filename=None, features=None, targets=None,
                 with_header=False, delimiter=',', targets_col=-1, **kwargs):
        if filename:
            if os.path.splitext(filename)[1] == '.csv':
                # Lire le fichier CSV entier dans une trame-de-données Pandas
                if with_header:
                    data_df = pd.read_csv(filename, delimiter=delimiter)
                else:
                    data_df = pd.read_csv(filename, delimiter=delimiter, header=None)

                # Extract the data from the remaining columns of the dataframe
                self.features = data_df.drop(data_df.columns[targets_col], axis=1)

                # Extract the targets from the last column of the dataframe
                self.targets = data_df.iloc[:, targets_col] - kwargs['offset']
            else:
                return 'Format inconnu !'
        else:
            if features and targets:
                self.features = features
                self.targets = targets

            else:
                return 'Données d\'entrée introuvables'

    def split_data(self, test_size=0.2, rs=123):
        train, true, train_targets, true_targets = train_test_split(self.features, self.targets, test_size=test_size,
                                                                    random_state=rs)
        return (train, train_targets), (true, true_targets)

    def __len__(self):
        return len(self.features), len(self.targets)


class CustomDataset(Dataset):
    def __init__(self, inputData, transform=None, target_transform=None):
        if isinstance(inputData, tuple):
            assert len(inputData) == 2
            features = inputData[0]
            targets = inputData[1]
        elif isinstance(inputData, dict):
            pass
        else:
            return 'format de données inconnu'

        self.n_classes = len(np.unique(targets))
        self.n_features = features.shape[-1]

        self.transform = transform
        self.target_transform = target_transform
        if isinstance(features, np.ndarray):
            self.data = torch.tensor(features).float()
        else:
            self.data = torch.tensor(features.values).float()
        if isinstance(targets, np.ndarray):
            self.targets = torch.tensor(targets).long()
        else:
            self.targets = torch.tensor(targets.values).long()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.transform(y)

        return x, y


class DatasetTransforms:
    def __init__(self):
        # Définir les transformateurs des données
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __call__(self, sample):
        # Appliquer le transformateur.
        data = sample['data']
        data = self.transforms(data)
        sample['data'] = data
        return sample
