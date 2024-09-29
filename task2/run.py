# # -*- coding: utf-8 -*-
# """
# Created on Tue Nov 28 21:00:28 2023

# @author: zheng
# """

# #pip install biosppy neurokit2 tsfel emd sktime tensorflow

import csv
import os

import biosppy.signals.ecg as ecg
import neurokit2 as nk
import tsfel
import emd
import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import tensorflow as tf
import IPython.display as ipd
# import matplotlib.pyplot as plt
# import scipy as sp
# from numpy.fft import rfft, irfft
# from tqdm import tqdm
# import pywt

# import keras

# from tensorflow.keras.layers import Dense, Flatten, Dropout
# from tensorflow.keras.models import Sequential

# from sklearn.impute import KNNImputer

# from sklearn.feature_selection import SelectKBest, chi2
# from scipy import stats
# from statistics import pstdev,variance
# from sklearn.preprocessing import normalize

# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import f1_score

# from sklearn.model_selection import KFold
# from sklearn.metrics import balanced_accuracy_score

# from sktime.transformations.panel.rocket import MiniRocket, MiniRocketMultivariateVariable

# from sklearn.model_selection import GridSearchCV, train_test_split
# from sklearn.pipeline import Pipeline
# from sklearn.feature_selection import SelectKBest, f_classif
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, make_scorer, f1_score
# import lightgbm as lgb
# import xgboost as xgb

# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.feature_selection import SelectKBest, f_classif

# from sklearn.impute import SimpleImputer

# ###ResNet

# import tensorflow as tf
# from tensorflow.keras.layers import Input, Dense, Conv1D, BatchNormalization, ReLU, Dropout, Bidirectional, LSTM
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.utils import to_categorical

# X_train = pd.read_csv("data/X_train.csv").drop('id', axis=1)
# y_train = pd.read_csv("data/y_train.csv").drop('id', axis=1)
# X_test = pd.read_csv("data/X_test.csv").drop('id', axis=1)


# n_cols = X_train.shape[1]
# to_pad = n_cols
# new_seq = []
# for one_seq in np.array(X_train):
#     one_seq = one_seq[~np.isnan(one_seq)]
#     len_one_seq = len(one_seq)
#     last_val = one_seq[-1]
#     n = to_pad - len_one_seq
#     to_concat = np.repeat(0, n)
#     new_one_seq = np.concatenate([one_seq, to_concat])
#     new_seq.append(new_one_seq)
# X_train_padded = np.stack(new_seq)
# #X_train_padded.shape

# n_cols = X_test.shape[1]
# to_pad = n_cols
# new_seq = []
# for one_seq in np.array(X_test):
#     one_seq = one_seq[~np.isnan(one_seq)]
#     len_one_seq = len(one_seq)
#     last_val = one_seq[-1]
#     n = to_pad - len_one_seq
#     to_concat = np.repeat(0, n)
#     new_one_seq = np.concatenate([one_seq, to_concat])
#     new_seq.append(new_one_seq)
# X_test_padded = np.stack(new_seq)
# #X_test_padded.shape

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import pandas as pd
# import numpy as np
# from torch.utils.data import DataLoader, TensorDataset
# import time

# # Residual Block
# class ResidualBlock(nn.Module):
#     def __init__(self, channels, kernel_size=5, stride=1, dropout_rate=0.5):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
#         self.bn1 = nn.BatchNorm1d(channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.dropout = nn.Dropout(dropout_rate)
#         self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
#         self.bn2 = nn.BatchNorm1d(channels)

#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.dropout(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out += residual  # Element-wise addition
#         out = self.relu(out)
#         return out

# # Simplified ResNet Model for ECG
# class ResNetECGSimplified(nn.Module):
#     def __init__(self, num_blocks=32, num_classes=128):
#         super(ResNetECGSimplified, self).__init__()
#         self.conv = nn.Conv1d(1, 64, kernel_size=7, stride=1, padding=3)
#         self.bn = nn.BatchNorm1d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.dropout = nn.Dropout(0.3)
#         self.blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_blocks)])
#         self.avg_pool = nn.AdaptiveAvgPool1d(1)
#         self.fc = nn.Linear(64, num_classes)

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.blocks(x)
#         x = self.avg_pool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x


# # Model instantiation
# model = ResNetECGSimplified()


# def dataframe_to_dataloader(df, labels=None, batch_size=32):
#     data_tensor = torch.tensor(df.values).float().unsqueeze(1)  # Add channel dimension
#     if labels is not None:
#         # If labels are one-hot encoded, convert them to class indices
#         if labels.ndim > 1:  # Assuming labels is a numpy array or a DataFrame
#             labels = np.argmax(labels, axis=1)
#         labels_tensor = torch.tensor(labels).long()
#         dataset = TensorDataset(data_tensor, labels_tensor)
#     else:
#         dataset = TensorDataset(data_tensor)
#     return DataLoader(dataset, batch_size=batch_size, shuffle=True if labels is not None else False)


# # Training function
# def train_model(model, train_loader, num_epochs=10):
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         start_time = time.time()
#         for i, (inputs, labels) in enumerate(train_loader):
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()

#         epoch_time = time.time() - start_time
#         print(f'Epoch: {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.3f}, Time: {epoch_time:.2f} sec')

#     print('Finished Training')


# # Model instantiation
# model = ResNetECGSimplified()

# # Assuming df_train, labels_train, and df_test are defined and contain your data
# df_train = pd.DataFrame(X_train_padded)
# labels_train = np.array(y_train)
# df_test = pd.DataFrame(X_test_padded)

# # Convert training DataFrame to DataLoader
# train_loader = dataframe_to_dataloader(df_train, labels_train)

# # Train the model
# train_model(model, train_loader)

# def extract_features(model, loader):
#     model.eval()
#     features = []
#     with torch.no_grad():
#         for batch in loader:
#             inputs = batch[0]  # Unpack the inputs; there are no labels
#             outputs = model(inputs)
#             features.append(outputs.cpu().numpy())
#     return pd.DataFrame(np.concatenate(features))

# # Convert testing DataFrame to DataLoader and extract features
# test_loader = dataframe_to_dataloader(df_test)
# train_features_df = extract_features(model, train_loader)
# test_features_df = extract_features(model, test_loader)

# # Display shapes of the feature DataFrames
# print("Training Features Shape:", train_features_df.shape)  # (num_train_samples, 128)
# print("Testing Features Shape:", test_features_df.shape)    # (num_test_samples, 128)

# train_features_df.columns = train_features_df.columns.astype(str)
# train_features_df.columns = train_features_df.columns.astype(str)

# pd.DataFrame(train_features_df).to_csv('BigResNet_train_features.csv', index=False)
# pd.DataFrame(test_features_df).to_csv('BigResNet_test_features.csv', index=False)