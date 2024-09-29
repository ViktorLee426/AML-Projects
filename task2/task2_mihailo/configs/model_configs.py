from util import make_serializable

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RationalQuadratic, ConstantKernel, RBF
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.ensemble import VotingClassifier, StackingClassifier

model_configs = {

    'default':{
        'model': LogisticRegression,
        'model_hyperparams': {},
        'param_grid': {}
    },

    # Optimize RandomForestClassifier
    'rf':{
        'model': RandomForestClassifier,
        'model_hyperparams': {},
        'param_grid': {}
    },

    'rf-basic':{
        'model': RandomForestClassifier,
        'model_hyperparams': {},
        'param_grid': {
            "model__n_estimators": [100, 200, 300, 400, 500, 600],
            "model__criterion": ["gini", "entropy", "log_loss"],
        }
    },

    'rf-md-mss':{
        'model': RandomForestClassifier,
        'model_hyperparams': {
            "n_estimators": 500,
            "criterion": "entropy",
        },
        'param_grid': {
            "model__max_depth": [2,4,8,16,32,64,128,256,None],
            "model__min_samples_split": [2,4,8,16,32,64,128,256,None],
        }
    },

    # Optimize HistGradientBoostingClassifier
    'hgb':{
        'model': HistGradientBoostingClassifier,
        'model_hyperparams': {},
        'param_grid': {}
    },

    'hgb_lr_mi':{
        'model': HistGradientBoostingClassifier,
        'model_hyperparams': {},
        'param_grid': {
            "model__learning_rate": [1e-3, 1e-2, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
            "model__max_iter": [50, 100, 200, 300]
        }
    },

    'hgb_lr_mi2':{
        'model': HistGradientBoostingClassifier,
        'model_hyperparams': {},
        'param_grid': {
            "model__learning_rate": [0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
            "model__max_iter": [300, 400, 500]
        }
    },

    'hgb_lr_mi3':{
        'model': HistGradientBoostingClassifier,
        'model_hyperparams': {},
        'param_grid': {
            "model__learning_rate": [1e-2, 5e-2, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
            "model__max_iter": [200, 300, 400]
        }
    },

    'hgb_mln_md':{
        'model': HistGradientBoostingClassifier,
        'model_hyperparams': {
            "learning_rate": 0.1,
            "max_iter": 300,
        },
        'param_grid': {
            "model__max_leaf_nodes": [8,16,32,64,128],
            "model__max_depth": [8,16,32,None]
        }
    },

    'hgb_opt':{
        'model': HistGradientBoostingClassifier,
        'model_hyperparams': {
            "learning_rate": 0.1,
            "max_iter": 300,
            "max_leaf_nodes": 128,
            "max_depth": 16,
        },
        'param_grid': {}
    },

    'hgb_es':{
        'model': HistGradientBoostingClassifier,
        'model_hyperparams': {
            "learning_rate": 0.1,
            "max_iter": 300,
            "max_leaf_nodes": 32,
            "max_depth": 64,
        },
        'param_grid': {
            "model__early_stopping": [True, False]
        }
    },

    'hgb_msl_l2':{
        'model': HistGradientBoostingClassifier,
        'model_hyperparams': {
            "learning_rate": 0.1,
            "max_iter": 300,
            "max_leaf_nodes": 32,
            "max_depth": 64,
        },
        'param_grid': {
            "model__min_samples_leaf": [2,4,8,16,32,64,128],
            "model__l2_regularization": [0.0, 0.2, 0.4, 0.8, 1.6, 3.2]
        }
    },

    # Optimize GradientBoostingClassifier
    'gb':{
        'model': GradientBoostingClassifier,
        'model_hyperparams': {},
        'param_grid': {}
    },

    # Optimize XGBClassifier
    'xgb':{
        'model': XGBClassifier,
        'model_hyperparams': {},
        'param_grid': {}
    },

    'xgb_g':{
        'model': XGBClassifier,
        'model_hyperparams': {},
        'param_grid': {
            "model__gamma": [0., 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]
        }
    },

    'xgb_ne_g':{
        'model': XGBClassifier,
        'model_hyperparams': {},
        'param_grid': {
            "model__n_estimators": [100, 200, 300],
            "model__gamma": [0., 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]
        }
    },

    'xgb_opt':{
        'model': XGBClassifier,
        'model_hyperparams': {
            "n_estimators": 300,
            "gamma": 0.0,
        },
        'param_grid': {}
    },

    'xgb_g16':{
        'model': XGBClassifier,
        'model_hyperparams': {
            'gamma': 1.6,
        },
        'param_grid': {}
    },

    'xgb_wide_search1':{
        'model': XGBClassifier,
        'model_hyperparams': {},
        'param_grid': {
            'model__n_estimators': [100, 150, 200, 250, 300, 350],
            'model__max_leaves': [3, 6, 12, 18, 24, 30, 35],
            'model__learning_rate':[1e-4, 1e-3, 1e-2, 5e-2, 5e-1, 1e-1]
        }
    },

    'xgb_best': {
        'model': XGBClassifier,
        'model_hyperparams': {
            'gamma': 0.0,
            "model__n_estimators": 100
        },
        'param_grid': {}
    },

    'xgb_wide_search2': {
        'model': XGBClassifier,
        'model_hyperparams': {
            'gamma': 0.0,
        },
        'param_grid': {
            'model__n_estimators': [100, 200, 300],
            'model__subsample' : [0.5, 0.7, 0.9, 1]
            }
    },

    'xgb_wide_search3': {
        'model': XGBClassifier,
        'model_hyperparams': {
            'gamma': 0.0,
        },
        'param_grid': {
            'model__n_estimators': [700, 900, 1200, 1500],
            'model__subsample' : [0.5, 0.7, 0.9]
            }
    },

    
    'xgb_wide_search4': {
        'model': XGBClassifier,
        'model_hyperparams': {
            'gamma': 0.0,
        },
        'param_grid': {
            'model__n_estimators': [700, 900, 1200, 1500],
            'model__subsample' : [0.5, 0.7, 0.9]
            }
    },

    # Optimize SVC
    'svc':{
        'model': SVC,
        'model_hyperparams': {},
        'param_grid': {}
    },

    'svc_rbf': {
        'model': SVC,
        'model_hyperparams': {
            'cache_size': 1
        },
        'param_grid': {
            "model__C": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100],
            "model__gamma": ['scale', 'auto', 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
        }
    },

    'svc_rbf2': {
        'model': SVC,
        'model_hyperparams': {
            'cache_size': 1
        },
        'param_grid': {
            "model__C": [0.7, 1, 3],
            "model__gamma": [1e-8, 1e-7, 1e-6]
        }
    },

    # 'svc_rbf10': {
    #     'model': SVC,
    #     'model_hyperparams': {
    #         'cache_size': 1
    #     },
    #     'param_grid': {
    #         "model__C": [0.2, 0.5, 0.7, 1],
    #         "model__"
    #     }
    # },

    # Optimize KNeighborsClassifier
    'knn':{
        'model': KNeighborsClassifier,
        'model_hyperparams': {},
        'param_grid': {}
    },

    'knn2':{
        'model': KNeighborsClassifier,
        'model_hyperparams': {},
        'param_grid': {
            'model__n_neighbors': [3, 5, 7, 10],#, 14, 18, 24, 30, 40, 50],
            'model__weights': ['distance'],
            'model__p': [1, 2]
        }
    },

    'knn3':{
        'model': KNeighborsClassifier,
        'model_hyperparams': {},
        'param_grid': {
            'model__n_neighbors': [14, 18, 24, 30, 40, 50],
            'model__weights': ['distance'],
            'model__p': [1, 2]
        }
    },

    'knn4':{
        'model': KNeighborsClassifier,
        'model_hyperparams': {},
        'param_grid': {
            'model__n_neighbors': [10, 11, 12, 13, 14, 15, 16, 17],
            'model__weights': ['distance'],
            'model__p': [1]
        }
    },

    'qda': {
        'model': QuadraticDiscriminantAnalysis,
        'model_hyperparams': {},
        'param_grid': {}
    },


    # Optimize LogisticRegression
    'log':{
        'model': LogisticRegression,
        'model_hyperparams': {},
        'param_grid': {}
    },

    'log2':{
        'model': LogisticRegression,
        'model_hyperparams': {},
        'param_grid': {
            'model__C': [0.01, 0.1, 0.5, 1, 1.5, 2, 10],
            'model__max_iter': [100, 200]
        }
    },

    'log3':{
        'model': LogisticRegression,
        'model_hyperparams': {
            'max_iter': 700
        },
        'param_grid': {
            'model__C': [0.01, 0.1, 0.5, 1, 1.5, 2, 10],
        }
    },

    'log4':{
        'model': LogisticRegression,
        'model_hyperparams': {
            'max_iter': 1500
        },
        'param_grid': {
            'model__C': [0.01, 0.03, 0.07, 0.1, 0.3, 0.7],
        }
    },

    # Optimize AdaBoostClassifier
    'ada':{
        'model': AdaBoostClassifier,
        'model_hyperparams': {},
        'param_grid': {}
    },

    # Optimize GaussianNB
    'gnb':{
        'model': GaussianNB,
        'model_hyperparams': {},
        'param_grid': {}
    },

    # Optimize MLPClassifier
    'mlp':{
        'model': MLPClassifier,
        'model_hyperparams': {},
        'param_grid': {}
    },

    # Optimize DecisionTreeClassifier
    'dt':{
        'model': DecisionTreeClassifier,
        'model_hyperparams': {},
        'param_grid': {}
    },

    'gp': {
        'model': GaussianProcessClassifier,
        'model_hyperparams': {
            'kernel': make_serializable(RBF(), 'RBF')# make_serializable(RationalQuadratic, "RatQuad"),
        },
        'param_grid': {}
    },

    'voting_xgb_log_knn': {
        'model': VotingClassifier,
        'model_hyperparams': {
            'estimators': [
                ('xgb', make_serializable(XGBClassifier(gamma=0.0, n_estimators=100) , 'XGBClassifier')),
                ('log', make_serializable(LogisticRegression(C=0.03, max_iter=1500), 'LogReg')),
                ('knn', make_serializable(KNeighborsClassifier(n_neighbors=14, p=1, weights='distance'), 'KNN'))   
            ]
        },
        'param_grid': {
            'model__voting': ['hard', 'soft']
        }
    },

    # {"model__max_depth": 32, "model__max_leaf_nodes": 64, learning_rate 0.1, max_iter=300}
    'stack_xgb_log_knn': {
        'model': StackingClassifier,
        'model_hyperparams': {
            'estimators': [
                # ('xgb', make_serializable(XGBClassifier(gamma=0.0, n_estimators=100) , 'XGBClassifier')),
                ('log', make_serializable(LogisticRegression(C=0.03, max_iter=1500), 'LogReg')),
                ('knn', make_serializable(KNeighborsClassifier(n_neighbors=14, p=1, weights='distance'), 'KNN')),
                # ('hgb', make_serializable(HistGradientBoostingClassifier(max_depth=32, max_leaf_nodes=64), 'HGB'))
            ],
            'final_estimator': make_serializable(XGBClassifier(gamma=0.0, n_estimators=100) , 'XGBClassifier'),
            # 'passthrough': True
        },
        'param_grid': {
            'model__passthrough': [True]
        }
    },

    'stack_hgb_log_knn': {
        'model': StackingClassifier,
        'model_hyperparams': {
            'estimators': [
                # ('xgb', make_serializable(XGBClassifier(gamma=0.0, n_estimators=100) , 'XGBClassifier')),
                ('log', make_serializable(LogisticRegression(C=0.03, max_iter=1500), 'LogReg')),
                ('knn', make_serializable(KNeighborsClassifier(n_neighbors=14, p=1, weights='distance'), 'KNN')),
                # ('hgb', make_serializable(HistGradientBoostingClassifier(max_depth=32, max_leaf_nodes=64), 'HGB'))
            ],
            'final_estimator': make_serializable(HistGradientBoostingClassifier(max_depth=32, 
                    max_leaf_nodes=64, learning_rate=0.1, max_iter=300), 'HGB'),
            # 'passthrough': True
        },
        'param_grid': {
            'model__passthrough': [True]
        }
    },
}