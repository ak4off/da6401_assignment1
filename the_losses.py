import numpy as np

class Losses:
    @staticmethod
    def cross_entropy(y_tru, y_pred):       #   -sum of y_true * log(y_pred) / no of egs
        y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
        return -np.sum(y_tru * np.log(y_pred)) / y_tru.shape[0]

    @staticmethod
    def mean_squared_error(y_tru, y_pred):      #   sumof [(y_true - y_pred) x 2] /total samples
        return np.mean((y_tru - y_pred)**2)

    @staticmethod
    def get(name):
        return getattr(Losses, name)
