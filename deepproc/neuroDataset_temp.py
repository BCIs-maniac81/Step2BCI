import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import Dataset

import torch
from torchvision import transforms


class CustomDataset(Dataset):
    def __init__(self, csv_file, targets_col=0, delimiter=',', test_size=0.2, rs=0, transform=None,
                 target_transform=None, with_header=True, **kwargs):
        # passer un tuple si les données sont déjà divisées en train et tester
        if len(csv_file) == 2:
            # Lire le fichier CSV de formation dans une trame-de-données Pandas
            if with_header:
                train_df = pd.read_csv(csv_file[0], delimiter=delimiter)
            else:
                train_df = pd.read_csv(csv_file[0], delimiter=delimiter, header=None)
            train = train_df.drop(train_df.columns[targets_col], axis=1)  # Supprimer la colonne cible

            train_targets = (train_df.iloc[:, targets_col]) - kwargs['offset']  # Extraire colonne cible

            self.n_classes = len(np.unique(train_targets))
            self.n_features = train.shape[-1]

            # Lire le fichier CSV de test dans une trame-de-données Pandas
            if with_header:
                test_df = pd.read_csv(csv_file[1], delimiter=delimiter)
            else:
                test_df = pd.read_csv(csv_file[1], delimiter=delimiter, header=None)

            test = test_df.drop(test_df.columns[targets_col], axis=1)
            test_targets = (test_df.iloc[:, targets_col]) - kwargs['offset']
            self.train = torch.tensor(train.values).float()
            self.train_targets = torch.tensor(train_targets.values).long()
            self.test = torch.tensor(test.values).float()
            self.test_targets = torch.tensor(test_targets.values).long()
        else:
            # Lire le fichier CSV entier dans une trame-de-données Pandas
            if with_header:
                data_df = pd.read_csv(csv_file, delimiter=delimiter)
            else:
                data_df = pd.read_csv(csv_file, delimiter=delimiter, header=None)

            # Extract the data from the remaining columns of the dataframe
            data = data_df.drop(data_df.columns[targets_col], axis=1)

            # Extract the targets from the last column of the dataframe
            targets = data_df.iloc[:, targets_col] - kwargs['offset']

            # Split the data and targets into train and test sets
            train, test, train_targets, test_targets = train_test_split(data, targets, test_size=test_size,
                                                                        random_state=rs)

            self.n_classes = len(np.unique(train_targets))
            self.n_features = train.shape[-1]

            self.train = torch.tensor(train.values).float()
            self.train_targets = torch.tensor(train_targets.values).long()
            self.test = torch.tensor(test.values).float()
            self.test_targets = torch.tensor(test_targets.values).long()

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.train), len(self.test)

    def __getitem__(self, index):
        x1 = self.train[index]
        x2 = self.test[index]
        y1 = self.train_targets[index]
        y2 = self.test_targets[index]

        if self.transform:
            x1 = self.transform(x1)
            x2 = self.transform(x2)
        if self.target_transform:
            y1 = self.transform(y1)
            y2 = self.transform(y2)

        return (x1, y1), (x2, y2)


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


if __name__ == '__main__':
    print('This is test section !!')
