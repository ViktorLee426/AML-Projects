from util import make_serializable

from feature_engineering.feature_engineering import BaseFeatureEngineer
from scaler.scaler import BaseScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, Normalizer
from sklearn.feature_selection import (SelectKBest, f_regression, 
    mutual_info_regression, )
    
preprocessing_configs = {
    'default':{
        'order': ['imputer'], 

        'imputer': SimpleImputer,
        'imputer_hyperparams': {
            'strategy': 'mean'
        },

        # 'scaler': BaseScaler,
        # 'scaler_hyperparams': {},

        # 'feature_engineer': BaseFeatureEngineer,
        # 'feature_engineer_hyperparams': {},

        # 'selector': BaseSelector,
        # 'selector_hyperparams': {},

        'param_grid': {
        }
    },

    'simp_stand':{
        'order': ['imputer', 'scaler'], 

        'imputer': SimpleImputer,
        'imputer_hyperparams': {
            'strategy': 'mean'
        },

        'scaler': StandardScaler,
        'scaler_hyperparams': {},

        'param_grid': {
        }
    },

    'simp_stand_kbest':{
        'order': ['imputer', 'scaler', 'selector'], 

        'imputer': SimpleImputer,
        'imputer_hyperparams': {
            'strategy': 'mean'
        },

        'scaler': StandardScaler,
        'scaler_hyperparams': {},

        'selector': SelectKBest,
        'selector_hyperparams': {
            'score_func': f_regression,
        },

        'param_grid': {
            'selector__k': [120, 160, 200, 240, 300, "all"],
        }
    },

}