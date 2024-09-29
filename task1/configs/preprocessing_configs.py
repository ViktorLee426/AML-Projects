from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectKBest, f_regression

from outlier_detectors import BaseOutlierDetector, EricsVotingDetector

preprocessing_configs = {
    'baseline':{
        # Default baseline preprocessing
        'order': ['outlier_detector', 'imputer', 'scaler', 'selector'], 

        'outlier_detector': BaseOutlierDetector,
        'outlier_detector_hyperparams': {
        },

        'imputer': SimpleImputer,
        'imputer_hyperparams': {
            'strategy': 'median'
        },

        'scaler': StandardScaler,
        'scaler_hyperparams': {},

        'selector': SelectKBest,
        'selector_hyperparams': {
            'score_func': f_regression,
            'k': 100,
        },


        'param_grid': {}
    },
    'ericvotingoutlierdetector':{
        # Eric's voting outlier detector
        'order': ['outlier_detector', 'imputer', 'scaler', 'selector'], 

        'outlier_detector': EricsVotingDetector,
        'outlier_detector_hyperparams': {
        },

        'imputer': SimpleImputer,
        'imputer_hyperparams': {
            'strategy': 'median'
        },

        'scaler': StandardScaler,
        'scaler_hyperparams': {},

        'selector': SelectKBest,
        'selector_hyperparams': {
            'score_func': f_regression,
            'k': 175,
        },


        'param_grid': {}
    },
}

