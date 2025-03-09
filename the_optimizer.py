import numpy as np

class Optimizer:
    def __init__(self, layers, config):
        self.layers = layers
        self.config = config

        # Initialize momentum and cache for both weights (w) and biases (b)
        self.m_w = [np.zeros_like(layer['w']) for layer in layers]
        self.v_w = [np.zeros_like(layer['w']) for layer in layers]

        self.m_b = [np.zeros_like(layer['b']) for layer in layers]
        self.v_b = [np.zeros_like(layer['b']) for layer in layers]

        self.t = 0  # time step for Adam/Nadam

    def update(self, grads):
        self.t += 1  # timestep increment

        # Gradient Clipping -  configurable
        clip_value = getattr(self.config, 'gradient_clip', None)
        if clip_value is not None:
            grads = [(np.clip(dw, -clip_value, clip_value), np.clip(db, -clip_value, clip_value)) 
                     for dw, db in grads]

        for i, layer in enumerate(self.layers):
            dw, db = grads[i]
        #  weight decay to gradients
            if self.config.weight_decay > 0:
                dw += self.config.weight_decay * layer['w']  # Weight decay term for weights
                # db += self.config.weight_decay * layer['b'] 

            if self.config.optimizer == 'sgd':
                layer['w'] -= self.config.learning_rate * dw
                layer['b'] -= self.config.learning_rate * db

            elif self.config.optimizer == 'momentum':
                self.m_w[i] = self.config.momentum * self.m_w[i] - self.config.learning_rate * dw
                self.m_b[i] = self.config.momentum * self.m_b[i] - self.config.learning_rate * db
                layer['w'] += self.m_w[i]
                layer['b'] += self.m_b[i]

            elif self.config.optimizer == 'nag':
                # Removed unused lookahead_w
                self.m_w[i] = self.config.momentum * self.m_w[i] - self.config.learning_rate * dw
                self.m_b[i] = self.config.momentum * self.m_b[i] - self.config.learning_rate * db
                layer['w'] += self.m_w[i]
                layer['b'] += self.m_b[i]

            elif self.config.optimizer == 'rmsprop':
                self.v_w[i] = self.config.beta2 * self.v_w[i] + (1 - self.config.beta2) * (dw ** 2)
                self.v_b[i] = self.config.beta2 * self.v_b[i] + (1 - self.config.beta2) * (db ** 2)

                layer['w'] -= (self.config.learning_rate * dw) / (np.sqrt(self.v_w[i]) + self.config.epsilon)
                layer['b'] -= (self.config.learning_rate * db) / (np.sqrt(self.v_b[i]) + self.config.epsilon)

            elif self.config.optimizer == 'adam':
                self.adam_update(i, layer, dw, db)

            elif self.config.optimizer == 'nadam':
                self.nadam_update(i, layer, dw, db)

            else:
                raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def adam_update(self, i, layer, dw, db):
        beta1, beta2, epsilon = self.config.beta1, self.config.beta2, self.config.epsilon

        # Weights (w)
        self.m_w[i] = beta1 * self.m_w[i] + (1 - beta1) * dw
        self.v_w[i] = beta2 * self.v_w[i] + (1 - beta2) * (dw ** 2)

        m_hat_w = self.m_w[i] / (1 - beta1 ** self.t)
        v_hat_w = self.v_w[i] / (1 - beta2 ** self.t)

        layer['w'] -= self.config.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + epsilon)

        # Biases (b)
        self.m_b[i] = beta1 * self.m_b[i] + (1 - beta1) * db
        self.v_b[i] = beta2 * self.v_b[i] + (1 - beta2) * (db ** 2)

        m_hat_b = self.m_b[i] / (1 - beta1 ** self.t)
        v_hat_b = self.v_b[i] / (1 - beta2 ** self.t)

        layer['b'] -= self.config.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + epsilon)

    def nadam_update(self, i, layer, dw, db):
        beta1, beta2, epsilon = self.config.beta1, self.config.beta2, self.config.epsilon

        # Weights (w)
        self.m_w[i] = beta1 * self.m_w[i] + (1 - beta1) * dw
        self.v_w[i] = beta2 * self.v_w[i] + (1 - beta2) * (dw ** 2)  

        m_hat_w = self.m_w[i] / (1 - beta1 ** self.t)
        v_hat_w = self.v_w[i] / (1 - beta2 ** self.t)

        m_nadam_w = (beta1 * m_hat_w + (1 - beta1) * dw / (1 - beta1 ** self.t))
        layer['w'] -= self.config.learning_rate * m_nadam_w / (np.sqrt(v_hat_w) + epsilon)

        # Biases (b)
        self.m_b[i] = beta1 * self.m_b[i] + (1 - beta1) * db
        self.v_b[i] = beta2 * self.v_b[i] + (1 - beta2) * (db ** 2) 

        m_hat_b = self.m_b[i] / (1 - beta1 ** self.t)
        v_hat_b = self.v_b[i] / (1 - beta2 ** self.t)

        m_nadam_b = (beta1 * m_hat_b + (1 - beta1) * db / (1 - beta1 ** self.t))
        layer['b'] -= self.config.learning_rate * m_nadam_b / (np.sqrt(v_hat_b) + epsilon)
