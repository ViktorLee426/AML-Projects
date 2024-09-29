import numpy as np
import pandas as pd

class BaseOutlierDetector:
    def __init__(self):
        pass
    
    def fit(self, X, y):
        return self

    def transform(self, X):
        return X
    
    def drop(self, X, y):
        return X, y
    
class EricsVotingDetector:
    def __init__(self, n_estimators=100, n_jobs=-1):
        # Load pre-computed predictions in csv file task1/data/outliers_majority_vote_ocsvm_iforest_knn.csv
        # and store them in self.outlier_predictions
        self.outlier_predictions = pd.read_csv('data/outliers_majority_vote_ocsvm_iforest_knn.csv', index_col='id')

    def fit(self, X, y):
        return self

    def drop(self, X, y):
        # remove outliers from X and y (eg. the id's where self.outlier_predictions['outlier'] == 1.0)
        X = X[self.outlier_predictions['outlier'] == 0.0]
        y = y[self.outlier_predictions['outlier'] == 0.0]
        return X, y
    