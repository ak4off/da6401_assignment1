import numpy as np

def one_hot_encode(labels, num_classes=10):
    return np.eye(num_classes)[labels]


'''

def one_hot_encode(labels, num_classes=10):
    # zero matrix of shape (len(labels), no. of classes)
    one_hot = np.zeros((len(labels), num_classes), dtype=int)
    
    # indices to 1
    for i, label in enumerate(labels):
        one_hot[i, label] = 1
    
    return one_hot
'''