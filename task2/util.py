import pandas as pd
import numpy as np
import os


def load_base_data():
    # print the path of this file
    basepath = os.path.realpath(__file__).split("util.py")[0]

    X_train = pd.read_csv(os.path.join(basepath, "data/X_train.csv"), index_col="id")
    y_train = pd.read_csv(os.path.join(basepath, "data/y_train.csv"), index_col="id")
    X_test = pd.read_csv(os.path.join(basepath, "data/X_test.csv"), index_col="id")
    return X_train, y_train, X_test
