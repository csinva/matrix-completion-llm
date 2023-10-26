from typing import List
import datasets
import numpy as np
from sklearn.model_selection import train_test_split
import imodels


def get_datasets_as_matrices(dataset_names: List[str]):
    '''Todo: make this return a pytorch dataloader.
    Also make separate version of this which preserves X, y separately
    '''
    mats_train = []
    mats_test = []
    for dataset_name in dataset_names:
        # load tabular data
        X_train, X_test, y_train, y_test, feature_names = imodels.get_clean_dataset(
            dataset_name, data_source='imodels', test_size=0.33)
        mat_train = np.concatenate([X_train, y_train.reshape(-1, 1)], axis=1)
        mat_test = np.concatenate([X_test, y_test.reshape(-1, 1)], axis=1)
        mats_train.append(mat_train)
        mats_test.append(mat_test)
    return mats_train, mats_test
