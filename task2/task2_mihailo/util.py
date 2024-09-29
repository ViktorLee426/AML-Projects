import numpy as np
import pandas as pd
import json
import numpy as np

def make_serializable(f, name):
    f.__name__ = name
    return f
    
def normalize_signal(signal):
    norm_signal = (signal - signal.min())/(signal.max() - signal.min())
    return norm_signal

def standardize_signal(signal):
    stand_signal = (signal - signal.mean())/(signal.std())
    return stand_signal

def load_base_data(data_path = "../data", just_train=False, flip=True, normal=False, standard=False):
    assert not (normal == True and standard == True), "Cannot normalize and standardize data at the same time."

    X_train = pd.read_csv(f'{data_path}/base/X_train.csv', index_col='id')
    y_train = pd.read_csv(f'{data_path}/base/y_train.csv', index_col='id')

    if flip:
        with open('../outliers/train/flipped.json') as json_file:
            flipped_indices = json.load(json_file)
        X_train = flip_df(X_train, flipped_indices)
    
    if normal:
        X_train = normalize_df(X_train)
    elif standard:
        X_train = standardize_df(X_train)
    
    if not just_train:
        X_test = pd.read_csv(f'{data_path}/base/X_test.csv', index_col='id')
        if flip:
            with open('../outliers/test/flipped.json') as json_file:
                flipped_indices = json.load(json_file)
            X_test = flip_df(X_test, flipped_indices)

            if normal:
                X_test = normalize_df(X_test)
            elif standard:
                X_test = standardize_df(X_test)

        return X_train, y_train, X_test
    else:
        return X_train, y_train

def normalize_df(df):
    df_normed = df.sub(df.min(axis=1),axis=0).divide(df.max(axis=1) - df.min(axis=1), axis=0)
    return df_normed

def standardize_df(df):
    df_standardized = df.sub(df.mean(axis=1),axis=0).divide(df.std(axis=1), axis=0)
    return df_standardized

def hist_statistic_per_class(statistic, y):
    """statistic = df containing the statistic (1 value per signal)"""
    statistic.index = y.values.flatten()
    statistic.index.name = 'class'
    statistic.groupby('class').hist(density=True, alpha=0.6, legend=True)
    
def load_extracted_dataset(dataset_name):
    """
    dataset_name: name of the folder in data
    """
    X_train = pd.read_csv('../data/' + dataset_name + '/X_train.csv', index_col='id')
    X_test = pd.read_csv('../data/' + dataset_name + '/X_test.csv', index_col='id')
    y_train = pd.read_csv('../data/base/y_train.csv', index_col='id')
    return X_train, y_train, X_test

def load_datasets_concat(dataset_names, data_path = "../data", features_json=None):
    y_train = pd.read_csv(f'{data_path}/base/y_train.csv', index_col='id')
    X_train_dfs, X_test_dfs = [], []
    for dn in dataset_names:
        X_train_dfs.append(pd.read_csv(f'{data_path}/{dn}/X_train.csv', index_col='id').add_prefix(dn + '/'))
        X_test_dfs.append(pd.read_csv(f'{data_path}/{dn}/X_test.csv', index_col='id').add_prefix(dn + '/'))
    concated_X_train = pd.concat(X_train_dfs, axis=1)
    concated_X_test = pd.concat(X_test_dfs, axis=1)
    replace_infinities(concated_X_train, concated_X_test)

    if features_json is not None:
        selected_features = load_json(f'feature_selection/{features_json}')
        all_features = concated_X_train.columns
        for fn in all_features:
            if fn not in selected_features:
               concated_X_train.drop(columns=fn, inplace=True) 
               concated_X_test.drop(columns=fn, inplace=True) 
    return concated_X_train, y_train, concated_X_test

def flip_df(df, flipped_indices):
    df_copy = df.copy()
    df_copy.iloc[flipped_indices] = -1 * df_copy.iloc[flipped_indices]
    return df_copy

def clean_nans(dirty):
    cleaned = np.array(dirty)[~np.isnan(dirty)]
    return cleaned


def write_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)

def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
        return data

def replace_infinities(X_train, X_test):
    X_train.replace([np.inf], 1000, inplace=True)
    X_train.replace([-np.inf], -1000, inplace=True)
    X_test.replace([np.inf], 1000, inplace=True)
    X_test.replace([-np.inf], -1000, inplace=True)

    X_train.dropna(axis=1, how='all')