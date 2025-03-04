import numpy as np

def accuracy(y_tru, y_pred):
    return np.mean(y_tru == y_pred)

def confusion_matrix(y_tru, y_pred, num_classes=10):
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_tru, y_pred):
        matrix[t, p] += 1
    return matrix
