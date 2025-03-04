import numpy as np

class Optimizer:
    def __init__(self, layers, config):
        self.layers = layers
        self.config = config

        # momentum and cache for all layers initialising 
        self.m = [np.zeros_like(layer['w']) for layer in layers]
        self.v = [np.zeros_like(layer['w']) for layer in layers]
        self.t = 0  # time step for Adam/Nadam
    
    def update(self, grads):
        self.t += 1     # timestep

        # gradient clipping (varying clip value)
        clip_value = 1.0
        grads = [np.clip(g, -clip_value, clip_value) for g in grads]

        for i, layer in enumerate(self.layers):
            if self.config.optimizer == 'sgd':      #vanilla grad desc
                layer['w'] -= self.config.learning_rate * grads[i]

            elif self.config.optimizer == 'momentum':   #mom based grad desc
                self.m[i] = self.config.momentum * self.m[i] - self.config.learning_rate * grads[i]
                layer['w'] += self.m[i]

            elif self.config.optimizer == 'nag':        #nostrov
                lookahead_w = layer['w'] + self.config.momentum * self.m[i]
                layer['w'] -= self.config.learning_rate * grads[i]
                self.m[i] = self.config.momentum * self.m[i] - self.config.learning_rate * grads[i]

            elif self.config.optimizer == 'rmsprop':
                self.v[i] = self.config.beta2 * self.v[i] + (1 - self.config.beta2) * (grads[i] ** 2)
                layer['w'] -= (self.config.learning_rate * grads[i]) / (np.sqrt(self.v[i]) + self.config.epsilon)

            elif self.config.optimizer == 'adam':
                self.adam_update(i, layer, grads[i])

            elif self.config.optimizer == 'nadam':      #combin of Adam and Nesterov
                self.nadam_update(i, layer, grads[i])

            else:
                raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    
    # no grad clipping
    # def update(self, grads):
    #     self.t += 1
    #     for i, layer in enumerate(self.layers):
    #         if self.config.optimizer == 'sgd':
    #             layer['w'] -= self.config.learning_rate * grads[i]

    #         elif self.config.optimizer == 'momentum':
    #             self.m[i] = self.config.momentum * self.m[i] - self.config.learning_rate * grads[i]
    #             layer['w'] += self.m[i]

    #         elif self.config.optimizer == 'nag':
    #             lookahead_w = layer['w'] + self.config.momentum * self.m[i]
    #             layer['w'] -= self.config.learning_rate * grads[i]
    #             self.m[i] = self.config.momentum * self.m[i] - self.config.learning_rate * grads[i]

    #         elif self.config.optimizer == 'rmsprop':
    #             self.v[i] = self.config.beta2 * self.v[i] + (1 - self.config.beta2) * (grads[i] ** 2)
    #             layer['w'] -= (self.config.learning_rate * grads[i]) / (np.sqrt(self.v[i]) + self.config.epsilon)

    #         elif self.config.optimizer == 'adam':
    #             self.adam_update(i, layer, grads[i])

    #         elif self.config.optimizer == 'nadam':
    #             self.nadam_update(i, layer, grads[i])

    #         else:
    #             raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def adam_update(self, i, layer, grad):
        beta1, beta2, epsilon = self.config.beta1, self.config.beta2, self.config.epsilon

        self.m[i] = beta1 * self.m[i] + (1 - beta1) * grad
        self.v[i] = beta2 * self.v[i] + (1 - beta2) * (grad ** 2)

        m_hat = self.m[i] / (1 - beta1 ** self.t)
        v_hat = self.v[i] / (1 - beta2 ** self.t)

        layer['w'] -= self.config.learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    def nadam_update(self, i, layer, grad):
        beta1, beta2, epsilon = self.config.beta1, self.config.beta2, self.config.epsilon

        self.m[i] = beta1 * self.m[i] + (1 - beta1) * grad
        m_hat = self.m[i] / (1 - beta1 ** self.t)

        v_hat = self.v[i] / (1 - beta2 ** self.t)
        m_nadam = (beta1 * m_hat + (1 - beta1) * grad / (1 - beta1 ** self.t))

        layer['w'] -= self.config.learning_rate * m_nadam / (np.sqrt(v_hat) + epsilon)


