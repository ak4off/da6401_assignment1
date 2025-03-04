import numpy as np

class Activations:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))         #   1/1+e^-x;

    @staticmethod
    def sigmoid_derivative(x):              #   sigmoid(x) * (1 - sigmoid(x))
        return x * (1 - x)

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)

    @staticmethod
    def tanh(x):
        return np.tanh(x)       # tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x)

    @staticmethod
    def tanh_derivative(x):     #1 - tanh(x)^2
        return 1 - np.tanh(x)**2

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)     #  make it a probability distribution

    @staticmethod
    def identity(x):        #   returns the input as is
        return x

    @staticmethod
    def identity_derivative(x):
        return np.ones_like(x)      #  derivative of the identity function is always 1

    @staticmethod
    def get(name):
        return getattr(Activations, name)

    @staticmethod
    def get_derivative(name):
        return getattr(Activations, f"{name}_derivative")
