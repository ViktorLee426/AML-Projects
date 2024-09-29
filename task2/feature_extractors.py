import numpy as np
import pandas as pd
import os


class BaseFeatureExtractor:
    def __init__(self):
        basepath = os.path.realpath(__file__).split("feature_extractors.py")[0]
        self.X_train = pd.read_csv(
            os.path.join(basepath, "data/X_train.csv"),
            index_col="id",
        )
        self.X_test = pd.read_csv(
            os.path.join(basepath, "data/X_test.csv"),
            index_col="id",
        )
        self.y_train = pd.read_csv(
            os.path.join(basepath, "data/y_train.csv"),
            index_col="id",
        )

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X

    def load_data(self):
        return self.X_train, self.y_train, self.X_test


class EricsBiobss:
    def __init__(self, n_estimators=100, n_jobs=-1):
        # Load pre-computed predictions in csv file task1/data/outliers_majority_vote_ocsvm_iforest_knn.csv
        # and store them in self.outlier_predictions
        basepath = os.path.realpath(__file__).split("feature_extractors.py")[0]
        self.X_train = pd.read_csv(
            os.path.join(basepath, "data/feature_extraction/bioss_X_train.csv"),
            index_col="id",
        )
        self.X_test = pd.read_csv(
            os.path.join(basepath, "data/feature_extraction/bioss_X_test.csv"),
            index_col="id",
        )
        self.y_train = pd.read_csv(
            os.path.join(basepath, "data/feature_extraction/bioss_y_train.csv"),
            index_col="id",
        )

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X

    def load_data(self):
        # remove outliers from X and y (eg. the id's where self.outlier_predictions['outlier'] == 1.0)
        return self.X_train, self.y_train, self.X_test


class EricsNeurokit:
    def __init__(self, n_estimators=100, n_jobs=-1):
        # Load pre-computed predictions in csv file task1/data/outliers_majority_vote_ocsvm_iforest_knn.csv
        # and store them in self.outlier_predictions
        basepath = os.path.realpath(__file__).split("feature_extractors.py")[0]
        self.X_train = pd.read_csv(
            os.path.join(basepath, "data/feature_extraction/neurokit2_X_train.csv"),
            index_col="id",
        )
        self.X_test = pd.read_csv(
            os.path.join(basepath, "data/feature_extraction/neurokit2_X_test.csv"),
            index_col="id",
        )
        self.y_train = pd.read_csv(
            os.path.join(basepath, "data/feature_extraction/neurokit2_y_train.csv"),
            index_col="id",
        )

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X

    def load_data(self):
        # remove outliers from X and y (eg. the id's where self.outlier_predictions['outlier'] == 1.0)
        return self.X_train, self.y_train, self.X_test


# class EricsManualExtraction:
#     def __init__(self, n_estimators=100, n_jobs=-1):
#         # Load pre-computed predictions in csv file task1/data/outliers_majority_vote_ocsvm_iforest_knn.csv
#         # and store them in self.outlier_predictions
#         basepath = os.path.realpath(__file__).split("feature_extractors.py")[0]
#         self.X_train = pd.read_csv(
#             os.path.join(basepath, "data/feature_extraction/train_features_manual.csv"),
#             index_col="id",
#         )
#         self.X_test = pd.read_csv(
#             os.path.join(basepath, "data/feature_extraction/test_features_manual.csv"),
#             index_col="id",
#         )
#         self.y_train = pd.read_csv(
#             os.path.join(basepath, "data/feature_extraction/neurokit2_y_train.csv"),
#             index_col="id",
#         )

#     def fit(self, X, y):
#         return self

#     def transform(self, X):
#         return X

#     def load_data(self):
#         # remove outliers from X and y (eg. the id's where self.outlier_predictions['outlier'] == 1.0)
#         return self.X_train, self.y_train, self.X_test


class EricsSpectral:
    def __init__(self, n_estimators=100, n_jobs=-1):
        # Load pre-computed predictions in csv file task1/data/outliers_majority_vote_ocsvm_iforest_knn.csv
        # and store them in self.outlier_predictions
        basepath = os.path.realpath(__file__).split("feature_extractors.py")[0]
        self.X_train = pd.read_csv(
            os.path.join(
                basepath,
                "data/feature_extraction/spectral_analysis_X_train_features.csv",
            ),
            index_col="id",
        )
        self.X_test = pd.read_csv(
            os.path.join(
                basepath,
                "data/feature_extraction/spectral_analysis_X_test_features.csv",
            ),
            index_col="id",
        )
        self.y_train = pd.read_csv(
            os.path.join(
                basepath, "data/feature_extraction/spectral_analysis_y_train.csv"
            ),
            index_col="id",
        )

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X

    def load_data(self):
        # remove outliers from X and y (eg. the id's where self.outlier_predictions['outlier'] == 1.0)
        return self.X_train, self.y_train, self.X_test


class EricsHRV:
    def __init__(self, n_estimators=100, n_jobs=-1):
        # Load pre-computed predictions in csv file task1/data/outliers_majority_vote_ocsvm_iforest_knn.csv
        # and store them in self.outlier_predictions
        basepath = os.path.realpath(__file__).split("feature_extractors.py")[0]
        self.X_train = pd.read_csv(
            os.path.join(basepath, "data/feature_extraction/hrv_X_train.csv"),
            index_col="id",
        )
        self.X_test = pd.read_csv(
            os.path.join(basepath, "data/feature_extraction/hrv_X_test.csv"),
            index_col="id",
        )
        self.y_train = pd.read_csv(
            os.path.join(basepath, "data/feature_extraction/hrv_y_train.csv"),
            index_col="id",
        )

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X

    def load_data(self):
        # remove outliers from X and y (eg. the id's where self.outlier_predictions['outlier'] == 1.0)
        return self.X_train, self.y_train, self.X_test


class EricsCombined:
    def __init__(self, n_estimators=100, n_jobs=-1):
        # Load pre-computed predictions in csv file task1/data/outliers_majority_vote_ocsvm_iforest_knn.csv
        # and store them in self.outlier_predictions
        basepath = os.path.realpath(__file__).split("feature_extractors.py")[0]
        self.X_train = pd.read_csv(
            os.path.join(basepath, "data/feature_extraction/combined_X_train.csv"),
            index_col="id",
        )
        self.X_test = pd.read_csv(
            os.path.join(basepath, "data/feature_extraction/combined_X_test.csv"),
            index_col="id",
        )
        self.y_train = pd.read_csv(
            os.path.join(basepath, "data/feature_extraction/combined_y_train.csv"),
            index_col="id",
        )

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X

    def load_data(self):
        # remove outliers from X and y (eg. the id's where self.outlier_predictions['outlier'] == 1.0)
        return self.X_train, self.y_train, self.X_test

class ViktorCombined:
    def __init__(self, n_estimators=100, n_jobs=-1):
        # Load pre-computed predictions in csv file task1/data/outliers_majority_vote_ocsvm_iforest_knn.csv
        # and store them in self.outlier_predictions
        basepath = os.path.realpath(__file__).split("feature_extractors.py")[0]
        self.X_train = pd.read_csv(
            os.path.join(basepath, "data/feature_extraction/Viktor_total_feature_train.csv"),
            index_col="id",
        )
        self.X_test = pd.read_csv(
            os.path.join(basepath, "data/feature_extraction/Viktor_total_feature_test.csv"),
            index_col="id",
        )
        self.y_train = pd.read_csv(
            os.path.join(basepath, "data/feature_extraction/combined_y_train.csv"),
            index_col="id",
        )

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X

    def load_data(self):
        # remove outliers from X and y (eg. the id's where self.outlier_predictions['outlier'] == 1.0)
        return self.X_train, self.y_train, self.X_test
