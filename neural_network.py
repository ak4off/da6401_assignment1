import numpy as np

class NeuralNetwork:
    def __init__(self, in_dim, out_dim, num_layers, hidden_size, activation, loss, optimizer,
                 learning_rate, momentum, beta, beta1, beta2, epsilon, weight_decay, weight_init,
                 epochs, batch_size, use_wandb=False):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.activation = activation
        self.loss = loss
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.weight_init = weight_init
        self.epochs = epochs
        self.batch_size = batch_size
        self.use_wandb = use_wandb

        # ✅ Initialize layers
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize input layer weights based on the weight initialization strategy"""
        np.random.seed(42)
        if self.weight_init == "random":
            self.weights = [np.random.randn(self.hidden_size, self.in_dim) * 0.01]
        elif self.weight_init == "Xavier":
            self.weights = [np.random.randn(self.hidden_size, self.in_dim) * np.sqrt(1. / self.in_dim)]

    def run(self, xtrain, ytrain, xvalid, yvalid):
        print("Training Neural Network...")

    def test(self, xtest, ytest):
        print("Testing Neural Network...")
        return np.zeros_like(ytest)  # Placeholder output
