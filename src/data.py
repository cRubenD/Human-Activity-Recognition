import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset

"""
Cerin?a 1: S? se creeze/caute un dataset pentru un anumit task, tip tabele, instan?e cu mai mul?i parametri + clase
"""


class HAR_Dataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, scaler: StandardScaler = None):
        if scaler is not None:
            features = scaler.transform(features)
        self.X = torch.from_numpy(features).float()
        self.y = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_har_data(data_dir):
    """
    Load train.csv and test.csv, encode labels and return arrays + class names.
    """
    train_csv = os.path.join(data_dir, 'train.csv')
    test_csv = os.path.join(data_dir, 'test.csv')

    if os.path.exists(train_csv) and os.path.exists(test_csv):
        df_train = pd.read_csv(train_csv)
        df_test = pd.read_csv(test_csv)
        le = LabelEncoder()
        y_train = le.fit_transform(df_train['Activity'])
        y_test = le.transform(df_test['Activity'])

        activity_names = le.classes_

        # drop the original Activity column
        X_train = df_train.drop(columns=['Activity']).values
        X_test = df_test.drop(columns=['Activity']).values
        return X_train, y_train, X_test, y_test, activity_names
    raise FileNotFoundError(f"No HAR data found in {data_dir}")
