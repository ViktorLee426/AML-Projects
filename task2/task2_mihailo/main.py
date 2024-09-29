import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer

from configs.model_configs import model_configs
from configs.preprocessing_configs import preprocessing_configs

from util import load_datasets_concat, replace_infinities

from pdb import set_trace

import warnings
warnings.filterwarnings("ignore")

np.random.seed(3141592)

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-pc', '--preprocessing_config', required=True, type=str,
                        help="Name of the preprocessing config in configs/preprocessing_configs "\
                            "used for this experiment.")
    parser.add_argument('-mc', '--model_config', required=True, type=str,
                        help="Name of the model config in configs/model_configs "\
                            "used for this experiment.")
    parser.add_argument('-ft', '--features', type=str, default=None, help="Path to features json.")
    parser.add_argument('-cv', '--cv_folds', type=int, help="Number of folds to use in CV. Default is 5.")
    parser.add_argument('-nj', '--n_jobs', type=int, default=1, 
                        help="Number of jobs used for grid search. More than 1 uses multithreading.")
    parser.add_argument('-data', '--data', nargs='+', required=True, help="List of directory names of the datasets in data/")
    args = parser.parse_args()

    if args.preprocessing_config not in preprocessing_configs:
        raise ValueError(f"Unkown preprocessing config {args.preprocessing_config}.")
    if args.model_config not in model_configs:
        raise ValueError(f"Unkown model config {args.model_config}.")

    experiment_name = f"{args.preprocessing_config}-{args.model_config}"
    experiment_name = experiment_name.split('.')[0]
    if os.path.isdir(f"results/{experiment_name}"):
        raise Exception(f"Results for experiment {experiment_name} already exist. Delete the " \
                        f"results/{experiment_name} folder to be able to redo the experiment.")

    p_config = preprocessing_configs[args.preprocessing_config]
    m_config = model_configs[args.model_config]

    # Load the data
    X_train, y_train, X_test = load_datasets_concat(args.data, data_path="data", features_json=args.features)
    replace_infinities(X_train, X_test)
    print(X_train.shape)

    # Feature scaling
    if 'scaler' in p_config['order']:
        scaler = p_config['scaler'](**p_config['scaler_hyperparams'])
    
    # Feature selection
    if 'selector' in p_config['order']:
        selector = p_config['selector'](**p_config['selector_hyperparams'])

    # Feature engineering
    if 'feature_engineer' in p_config['order']:
        feature_engineer = p_config['feature_engineer'](**p_config['feature_engineer_hyperparams'])

    # Model
    model = m_config['model'](**m_config['model_hyperparams'])

    # Pipeline
    def get_pipeline_step(key):
        return (key, p_config[f'{key}'](**p_config[f'{key}_hyperparams']))
    
    steps = [get_pipeline_step(step) for step in p_config['order']] + [('model', model)]
    pipeline = Pipeline(steps=steps)

    # Grid Search CV
    param_grid = p_config['param_grid']
    param_grid.update(m_config['param_grid'])  # Combine param_grids from p_config and m_config
    grid_search = GridSearchCV(pipeline, param_grid, n_jobs=args.n_jobs, 
                                scoring=make_scorer(accuracy_score), return_train_score=True,
                                verbose=3, cv=args.cv_folds)
    grid_search.fit(X_train, np.ravel(y_train))

    # Collect all results
    print(f"Best parameters (CV score={grid_search.best_score_:0.3f}):")
    print(grid_search.best_params_)
    cv_results = grid_search.cv_results_
    cv_results['best_score'] = grid_search.best_score_
    cv_results['best_params'] = grid_search.best_params_

    # Make results json serializable
    for key in cv_results:
        if isinstance(cv_results[key], np.ndarray):
            cv_results[key] = cv_results[key].tolist()

    # Save the results
    data_name = "_".join(args.data)
    features_name = 'None' if args.features is None else args.features.split('.')[0]
    experiment_name = f'{experiment_name}-{features_name}'
    print(f"Saving results to results/{data_name}/{experiment_name}")
    path = Path(f'results/{data_name}/{experiment_name}')
    path.mkdir(parents=True, exist_ok=True)
    config = {
        'preprocessing_config': p_config,
        'model_config': m_config,
        'n_jobs': args.n_jobs,
        'cv_folds': args.cv_folds,
        'data': data_name
    }
    with open(f'results/{data_name}/{experiment_name}/config.json', 'w') as f:
        json.dump(config, f, default=lambda f: f.__name__)
    with open(f'results/{data_name}/{experiment_name}/results.json', 'w') as f:
        json.dump(grid_search.cv_results_, f, default=lambda f: f.__name__)

    # Generate test set prediction
    y_test = grid_search.predict(X_test)
    y_test = pd.DataFrame(y_test, columns=['y'])
    y_test.to_csv(f'results/{data_name}/{experiment_name}/y_test.csv', index_label='id')

    # # Generate train set prediction
    # y_train_pred = grid_search.predict(X_train)
    # y_train_pred = pd.DataFrame(y_train_pred, columns=['y'])
    # y_train_pred.to_csv(f'results/{data_name}/{experiment_name}/y_train_pred.csv', index_label='id')