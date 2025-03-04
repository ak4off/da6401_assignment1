import numpy as np
import argparse
from the_activations import Activations
from the_losses import Losses
from the_optimizer import Optimizer
from the_data import one_hot_encode
from wandb_logger import WandbLogger
from calcc import accuracy

class NeuralNetwork:
    def __init__(self, **kwargs):
        self.config = argparse.Namespace(**kwargs)
        self.logger = WandbLogger(self.config.use_wandb)

        self.layers = []
        in_size = self.config.in_dim        #   input layer size

        for _ in range(self.config.num_layers):
            self.layers.append({'w': self.init_weights(in_size, self.config.hidden_size)})
            in_size = self.config.hidden_size

        self.layers.append({'w': self.init_weights(in_size, self.config.out_dim)})

        self.activation_fn = Activations.get(self.config.activation)
        self.activation_derivative = Activations.get_derivative(self.config.activation)
        self.loss_fn = Losses.get(self.config.loss)
        self.optimizer = Optimizer(self.layers, self.config)

    # def init_weights(self, in_size, out_size):
    #     if self.config.weight_init == 'Xavier':
    #         return np.random.randn(in_size, out_size) * np.sqrt(2 / (in_size + out_size))
    #     return np.random.randn(in_size, out_size) * 0.01
    def init_weights(self, in_size, out_size):
        if self.config.weight_init == 'Xavier':
            # xavier weight initialization (for tanh)
            return np.random.randn(in_size, out_size) * np.sqrt(2 / (in_size + out_size))
        elif self.config.weight_init == 'He':
            # He initialization (for ReLU) not in use
            return np.random.randn(in_size, out_size) * np.sqrt(2 / in_size)
        else:
            # random weight initialization (default)
            return np.random.randn(in_size, out_size) * 0.01

    def evaluate(self, x, y):
        _, y_pred = self.forward(x)
        y_one_hot = one_hot_encode(y, num_classes=10)

        #  loss using the given loss function
        loss = self.loss_fn(y_one_hot, y_pred)

        #  accuracy
        predicted_labels = np.argmax(y_pred, axis=1)
        accuracy = np.sum(predicted_labels == y) / len(y)

        return loss, accuracy


    def forward(self, x):
        activations = [x]       #   list of activations with the input
        for layer in self.layers[:-1]:
            x = self.activation_fn(x @ layer['w'])  # activation for all but o/p layer
            activations.append(x)

        logits = x @ self.layers[-1]['w']       #   o/p layer
        y_pred = Activations.softmax(logits)    
        return activations, y_pred

    def backward(self, activations, y_tru, y_pred):
        grads = []
        delta = y_pred - y_tru

        for i in reversed(range(len(self.layers))):
            grads.insert(0, activations[i].T @ delta / y_tru.shape[0])
            if i > 0:
                delta = (delta @ self.layers[i]['w'].T) * self.activation_derivative(activations[i])

        return grads

    def train_batch(self, x, y_tru):
        activations, y_pred = self.forward(x)
        grads = self.backward(activations, y_tru, y_pred)
        self.optimizer.update(grads)
        return self.loss_fn(y_tru, y_pred), y_pred

    #   training
    def run(self, train_img, train_labe, val_img, val_labe):
        num_samples = train_img.shape[0]
        num_batches = num_samples // self.config.batch_size

        for epoch in range(self.config.epochs):
            train_loss = 0.0
            correct_train = 0
            total_train = 0

            # shuffle training data at the start of every epoch
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            train_img = train_img[indices]
            train_labe = train_labe[indices]

            for i in range(num_batches):
                batch_start = i * self.config.batch_size
                batch_end = (i + 1) * self.config.batch_size

                x_batch = train_img[batch_start:batch_end]
                y_batch = train_labe[batch_start:batch_end]

                # one-hot rep
                y_batch_one_hot = one_hot_encode(y_batch, num_classes=10)

                # forward pass
                activations, y_pred = self.forward(x_batch)

                # loss 
                loss = self.loss_fn(y_batch_one_hot, y_pred)
                train_loss += loss

                # backprop and weight update
                grads = self.backward(activations, y_batch_one_hot, y_pred)
                self.optimizer.update(grads)

                # training accuracy
                predicted_labels = np.argmax(y_pred, axis=1)
                correct_train += np.sum(predicted_labels == y_batch)
                total_train += len(y_batch)

            # training loss
            train_loss /= num_batches
            train_accuracy = correct_train / total_train

            # validation loss & accuracy
            val_loss, val_accuracy = self.evaluate(val_img, val_labe)

            h
            print(f"Epoch [{epoch+1}/{self.config.epochs}]")
            print(f"    Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
            print(f"    Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

            # results to wandb
            if self.config.use_wandb:
                self.logger.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy
                })

    # def run(self, X_train, y_train, X_val, y_val):
    #     y_train = one_hot_encode(y_train)
    #     y_val = one_hot_encode(y_val)

    #     for epoch in range(self.config.epochs):
    #         losses = []
    #         for i in range(0, len(X_train), self.config.batch_size):
    #             x_batch = X_train[i:i+self.config.batch_size]
    #             y_batch = y_train[i:i+self.config.batch_size]
    #             loss, _ = self.train_batch(x_batch, y_batch)
    #             losses.append(loss)

    #         self.logger.log({"epoch": epoch, "loss": np.mean(losses)})


    #   testing the network
    def test(self, X, y):
        _, y_pred = self.forward(X) 
        return np.argmax(y_pred, axis=1)        #   class with highest prob
