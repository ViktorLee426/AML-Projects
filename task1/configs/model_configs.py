from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, RBF, ConstantKernel as C
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import BayesianRidge

import numpy as np

RANDOM_SEED = 3141592

model_configs = {
    "baseline": {
        # Default baseline model.
        "model": GradientBoostingRegressor,
        "model_hyperparams": {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "subsample": 0.45,
        },
        "param_grid": {},
    },
    "gbr_big_grid": {
        # GradientBoostingRegressor model.
        "model": GradientBoostingRegressor,
        "model_hyperparams": {
            "criterion": "squared_error", 
            "learning_rate": 0.05, 
            "max_depth": 8, 
            "max_features": 1.0,
            "min_samples_leaf": 1, 
            "min_samples_split": 0.1, 
            "n_estimators": 500, 
            "subsample": 0.7
        },
        "param_grid": {
            # "model__n_estimators": [100, 500, 1000, 2000],#grid3 from here
            # "model__learning_rate": [0.1,0.2],#[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
            # "model__subsample": [0.8,0.9,1.0],#[0.45, 0.5, 0.55, 0.7, 0.9, 1.0],
            # # "model__min_samples_split": np.linspace(0.1, 3, 12),#grid2 from here
            # # "model__min_samples_leaf": np.linspace(0.1, 2, 12),
            # # "model__max_depth":[3,5,8],
            # # "model__max_features":[0.5,0.8,0.9,0.95,1.0],# Grid1 from here
            # # "model__criterion": ["friedman_mse",  "squared_error"],
            # # "model__subsample":[0.4, 0.45, 0.5, 0.55, 0.6, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],

            # # Second try here with the best parameters from the first try
            # # "model__criterion": ["squared_error"],
            # # "model__max_features":[1.0],
            # # "model__max_depth":[6,8],
            # # "model__min_samples_split": [0.1,1,2],
            # # "model__min_samples_leaf": [0.1,0.5,1],
            # # "model__learning_rate": [0.1,0.05],
            # # "model__subsample": [0.7,0.8,0.9,1],
            # # "model__n_estimators": [500],
            # # Last gridsearch try 5 here
            # "model__criterion": ["squared_error"],
            # "model__max_features":[1.0],
            # "model__max_depth":[6,8],
            # "model__min_samples_split": [0.1,1,2],
            # "model__min_samples_leaf": [0.1,0.5,1],
            # "model__learning_rate": [0.1,0.05],
            # "model__subsample": [0.7,0.8,0.9,1.0],
            # "model__n_estimators": [500,600,700,1000],

        },
    },
    "catboost": {
        # Catboost model.
        "model": CatBoostRegressor,
        "model_hyperparams": {
            # "iterations": 200,
            # "learning_rate": 0.05,
            # "depth": 8,
            "eval_metric": "R2",
        },
        "param_grid": {
            "model__depth": [6, 8, 10],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__iterations": [30, 50, 100],
        },
    },
    "xgboost": {
        # Xgboost model.
        "model": XGBRegressor,
        "model_hyperparams": {},#"max_depth": 6, "eta": 0.3},
        "param_grid": {
            "model__max_depth": [3, 4, 5, 6, 8],
            "model__min_child_weight": [ 1, 3, 5, 7],
            "model__learning_rate": [0.05, 0.10, 0.15],
            "model__gamma":[ 0.0, 0.1, 0.2],
            "model__colsample_bytree":[ 0.3, 0.4],
        },
    },
    "adaboost": {
        # Adaboost model.
        "model": AdaBoostRegressor,
        "model_hyperparams": {
            "n_estimators": 550,
            "learning_rate": 1.1,
            "estimator": DecisionTreeRegressor(),
        },
        "param_grid": {
        #  "model__n_estimators": [500,550,600],
        #  "model__learning_rate":[0.8,0.9,1,1.1,1.2],
        #  "model__estimator": [DecisionTreeRegressor(), SVR(kernel=RationalQuadratic()), SGDRegressor(), BayesianRidge()],
        },
    },
    "lgbmregressor": {
        # LGBMRegressor model.
        "model": LGBMRegressor,
        "model_hyperparams": {
            "min_gain_to_split": 0.01,
        },
        "param_grid": {
            # "model__max_depth": [5, 6, 7, 8, 9],
            # "model__num_leaves": [32, 64, 70, 90, 100, 128, 140], # between 2^(max_depth-1) and 2^max_depth
            # "model__min_data_in_leaf": [10, 100, 500, 750, 1000, 1500, 2000], # to avoid overfitting
        },
    },
    "svr": {
        # SVR model.
        "model": SVR,
        "model_hyperparams": {},
        "param_grid": {},
    },
    "gpregressor": {
        # GaussianProcessRegressor model.
        "model": GaussianProcessRegressor,
        "model_hyperparams": {
                "kernel": RationalQuadratic(alpha=1, length_scale=1),
                "random_state": RANDOM_SEED,
                "n_restarts_optimizer": 5,
                "normalize_y": True,
        },
        "param_grid": {
            # # "model__n_restarts_optimizer": [1, 2, 3, 4, 5, 6, 8, 10, 14],
            # "model__normalize_y": [False, True],
            # "model__kernel": [RationalQuadratic(), RBF(), C(1.0, (1e-3, 1e3)) * RationalQuadratic()],
        },
    },
    "stackingregressor": {
        # StackingRegressor model.
        "model": StackingRegressor,
        "model_hyperparams": {
            "final_estimator": RidgeCV(),
            "cv": 5,
            "estimators": [
                ("gbr", GradientBoostingRegressor()),
                ("svr", SVR(kernel="linear")),
                ("catboost", CatBoostRegressor()),
                ("xgboost", XGBRegressor()),
                ("adaboost", AdaBoostRegressor()),
                ("lgbmregressor", LGBMRegressor()),
                ("gpregressor", GaussianProcessRegressor(kernel=RationalQuadratic())),
            ],
        },
        "param_grid": {},
    },
    "stackingregressor_with_learned_parameter": {
        # StackingRegressor model.
        "model": StackingRegressor,
        "model_hyperparams": {
            "final_estimator": RidgeCV(),
            "cv": 5,
            "estimators": [
                ("gbr", GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, subsample=0.45)),
                ("svr", SVR(kernel=RationalQuadratic())),
                ("catboost", CatBoostRegressor(depth=6, learning_rate=0.1, iterations=100)),
                ("xgboost", XGBRegressor(learning_rate=0.1, max_depth=6, min_child_weight=1)),
                ("adaboost", AdaBoostRegressor(n_estimators=550, learning_rate=1.1, estimator=DecisionTreeRegressor())),
                ("lgbmregressor", LGBMRegressor(min_gain_to_split=0.01, max_depth=6, num_leaves=45, min_data_in_leaf=10)),
                ("gpregressor", GaussianProcessRegressor(kernel=RationalQuadratic(alpha=1, length_scale=1), random_state=RANDOM_SEED, n_restarts_optimizer=5, normalize_y=True)),
            ],
        },
        "param_grid": {},
    },
}
