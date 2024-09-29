from pyod.models.ecod import ECOD
import pandas as pd
import numpy as np
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.knn import KNN
from sklearn.covariance import EllipticEnvelope
#from pyod.models.mcd import MCD
#from scipy import stats
from sklearn.cluster import DBSCAN
import math 
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression


class BaseOutlierDetector:
    # returned value = 1 for outliers, 0 for inliers
    def __init__(self):
        self.models = {
            'IForest': IForest(),
            'OCSVM': OCSVM(),
            'KNN': KNN(),
            'LOF': LOF(),
            'DBSCAN': DBSCAN()
        }

    def fit(self, X, n_detector=3):
        outlier_indicators = np.zeros(X.shape[0])#initialize a counter for each point

        for i, (model_name, model) in enumerate(self.models.items()):
            if i < n_detector:
                model.fit(X)
                if model_name == "LOF":
                    predictions = model.fit_predict(X)
                else:
                    predictions = model.predict(X)
                indices = [index for index, element in enumerate(predictions) if element == 1]
                outlier_indicators += (predictions == 1)

        # take the majority vote, i.e. the counter is greater than half of the number of detectors 
        returned_value = outlier_indicators >= math.ceil(n_detector/2)

        return [1  if value == True else 0 for value in returned_value]

    def fit_predict(self, X, n_detector=3):
        return self.fit(X, n_detector)

# Example usage:
if __name__ == "__main__":
    X_train = pd.read_csv("data/X_train.csv", index_col="id")
    y_train = pd.read_csv("data/y_train.csv", index_col="id").to_numpy().ravel()    
    
    pre_od_pipeline = Pipeline(
        [
            ("imputer", IterativeImputer(initial_strategy="median", n_nearest_features=5)),
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(score_func=f_regression, k=175)),
        ]
    )
    X_train_pre_od = pre_od_pipeline.fit_transform(X_train, y_train) 

    outlier_detector = BaseOutlierDetector()
    results = outlier_detector.fit_predict(X_train_pre_od, n_detector=3)
    print(results)
    res_indices = [index for index, element in enumerate(results) if element == 1]
    print(len(res_indices), res_indices)