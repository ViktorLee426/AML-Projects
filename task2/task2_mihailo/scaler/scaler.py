class BaseScaler:
    def __init__(self):
        pass
    
    def fit(self, X, y):
        return self

    def transform(self, X):
        return X