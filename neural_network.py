import numpy as np
import argparse
import wandb
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
        # self.use_wandb = self.config.use_wandb 

        self.layers = []
        in_size = self.config.in_dim        #   input layer size
        
        # Layer initialization (hidden layers + output layer)
        for _ in range(self.config.num_layers):
            self.layers.append({
                'w': self.init_weights(in_size, self.config.hidden_size),
                'b': np.zeros((1, self.config.hidden_size))  # Initialize biases to zeros
            })            
            in_size = self.config.hidden_size

        self.layers.append({
            'w': self.init_weights(in_size, self.config.out_dim),
            'b': np.zeros((1, self.config.out_dim))  # Output layer biases
        })        

        self.activation_fn = Activations.get(self.config.activation)
        self.activation_derivative = Activations.get_derivative(self.config.activation)
        self.loss_fn = Losses.get(self.config.loss)
        self.optimizer = Optimizer(self.layers, self.config)

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


    def evaluate(self, x, y_true):
        # y_pred = self.forward(x)[1]  # Get predictions
        _, y_pred = self.forward(x)
        y_one_hot = one_hot_encode(y_true, num_classes=10)  # Ensure labels are in one-hot form
        
        cross_entropy_loss = Losses.get('cross_entropy')(y_one_hot, y_pred)
        squared_error_loss = Losses.get('mean_squared_error')(y_one_hot, y_pred)  # FIXED
        
        accuracy = np.mean(np.argmax(y_pred, axis=1) == y_true)
        
        return cross_entropy_loss, squared_error_loss, accuracy


    def forward(self, x):
        activations = [x]       #   list of activations with the input
        for layer in self.layers[:-1]:
            x = self.activation_fn(x @ layer['w'] + layer['b'])  # Add + layer['b']

            activations.append(x)

        logits = x @ self.layers[-1]['w'] + self.layers[-1]['b']  # Add bias to final logits

        y_pred = Activations.softmax(logits)    
        return activations, y_pred

    def backward(self, activations, y_tru, y_pred):
        grads = []
        delta = y_pred - y_tru
        for i in reversed(range(len(self.layers))):
            dw = activations[i].T @ delta / y_tru.shape[0]   # Gradient w.r.t weights
            db = np.sum(delta, axis=0, keepdims=True) / y_tru.shape[0]  # Gradient w.r.t biases
            grads.insert(0, (dw, db))  # Store both weight and bias gradients

            if i > 0:
                delta = (delta @ self.layers[i]['w'].T) * self.activation_derivative(activations[i])

        return grads

    def train_batch(self, x, y_tru):
        activations, y_pred = self.forward(x)
        y_tru_one_hot = one_hot_encode(y_tru, num_classes=10)

        # Compute both loss functions
        cross_entropy_loss = Losses.get('cross_entropy')(y_tru_one_hot, y_pred)
        squared_error_loss = Losses.get('mean_squared_error')(y_tru_one_hot, y_pred)

        grads = self.backward(activations, y_tru_one_hot, y_pred)
        self.optimizer.update(grads)
        
        return cross_entropy_loss, squared_error_loss, y_pred

    #   training
    def run(self, train_img, train_labe, val_img, val_labe):

        # early stopping
        # best_val_accuracy = 0
        # best_train_accuracy = 0
        # early_stop_wait = 5
        # wait_count = 0

        train_cross_entropy_loss = 0.0  
        train_squared_error_loss = 0.0  

        if self.config.use_wandb:
            self.logger.log({
                "hidden_layer_size": self.config.hidden_size,
                "num_layers": self.config.num_layers,
                "activation": self.config.activation,
                "learning_rate": self.config.learning_rate,
                "optimizer": self.config.optimizer,
                "weight_decay": self.config.weight_decay,
            })
            
        num_samples = train_img.shape[0]
        num_batches = num_samples // self.config.batch_size

        val_accuracies = []
        train_accuracies = []
        train_losses = []
        val_losses = []
        
        train_cross_entropy_losses = []
        train_squared_error_losses = []
        val_cross_entropy_losses = []
        val_squared_error_losses = []
        
        for epoch in range(self.config.epochs):
            train_loss = 0.0
            correct_train = 0
            total_train = 0

            # Shuffle training data
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            train_img = train_img[indices]
            train_labe = train_labe[indices]

            for i in range(num_batches):
                batch_start = i * self.config.batch_size
                batch_end = (i + 1) * self.config.batch_size

                x_batch = train_img[batch_start:batch_end]
                y_batch = train_labe[batch_start:batch_end]

                # Forward and Backpropagation
                cross_entropy_loss, squared_error_loss, y_pred = self.train_batch(x_batch, y_batch)

                train_cross_entropy_loss += cross_entropy_loss
                train_squared_error_loss += squared_error_loss

                # Compute activations for backward pass
                # activations, _ = self.forward(x_batch)      # commnting cos already computed
                # y_batch_one_hot = one_hot_encode(y_batch, num_classes=10)
                # grads = self.backward(activations, y_batch_one_hot, y_pred)
                # self.optimizer.update(grads)

                # Training accuracy
                predicted_labels = np.argmax(y_pred, axis=1)
                correct_train += np.sum(predicted_labels == y_batch)
                total_train += len(y_batch)

                train_loss += cross_entropy_loss  # Accumulate loss over batches

            
            # Normalize training loss
            train_loss /= num_batches
            train_accuracy = correct_train / total_train

            # Validation loss & accuracy
            val_cross_entropy_loss, val_squared_error_loss, val_accuracy = self.evaluate(val_img, val_labe)

            # Store validation & training losses
            train_cross_entropy_losses.append(train_cross_entropy_loss / num_batches)
            train_squared_error_losses.append(train_squared_error_loss / num_batches)
            val_cross_entropy_losses.append(val_cross_entropy_loss)
            val_squared_error_losses.append(val_squared_error_loss)

            val_accuracies.append(val_accuracy)
            train_accuracies.append(train_accuracy)
            val_loss = (val_cross_entropy_loss + val_squared_error_loss) / 2
            val_losses.append(val_loss)

            # print(f"Epoch [{epoch+1}/{self.config.epochs}]")
            # print(f"    Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
            # print(f"    Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

            if self.config.use_wandb:
                self.logger.log({
                    "epoch": epoch + 1,
                    "train_cross_entropy_loss": train_cross_entropy_loss / num_batches,
                    "train_squared_error_loss": train_squared_error_loss / num_batches,
                    "val_cross_entropy_loss": val_cross_entropy_loss,
                    "val_squared_error_loss": val_squared_error_loss,
                    "train_accuracy": train_accuracy,
                    "val_accuracy": val_accuracy
                })

                # Log loss comparison plot
                wandb.log({
                    "Loss Comparison": wandb.plot.line_series(
                        xs=list(range(1, epoch+2)),
                        ys=[
                            train_cross_entropy_losses,
                            train_squared_error_losses,
                            val_cross_entropy_losses,
                            val_squared_error_losses
                        ],
                        keys=["Train CE Loss", "Train SE Loss", "Val CE Loss", "Val SE Loss"],
                        title="Cross-Entropy vs. Squared Error Loss",
                        xname="Epoch"
                    )
                })
        
            # if val_accuracy < best_val_accuracy or train_accuracy < best_train_accuracy:
            #     wait_count += 1
            #     print(f"Validation accuracy decreased. early_stop_wait counter: {wait_count}/{early_stop_wait}")
            #     if wait_count >= early_stop_wait:
            #         print("Early stopping triggered.")
            #         break
            # else:
            #     best_val_accuracy = max(best_val_accuracy, val_accuracy)
            #     best_train_accuracy = max(best_train_accuracy, train_accuracy)
            #     wait_count = 0

        # Run testing after training
        if self.config.use_wandb:
            # print("\n Running Test Evaluation...")
            test_loss, test_accuracy, _ = self.test(val_img, val_labe)
            # print(f"Final Test Accuracy: {test_accuracy:.4f}, Final Test Loss: {test_loss:.4f}")

    def test(self, X, y):
        _, y_pred = self.forward(X)
        y_pred_one_hot = Activations.softmax(y_pred)  # Ensure predictions are probabilities

        # Calculate the test loss
        y_one_hot = one_hot_encode(y, num_classes=10)
        test_loss = self.loss_fn(y_one_hot, y_pred_one_hot)

        # Calculate test accuracy
        predicted_labels = np.argmax(y_pred_one_hot, axis=1)
        test_accuracy = np.mean(predicted_labels == y)

        # print(f"Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")

        if self.config.use_wandb:
            wandb.log({"test_accuracy": test_accuracy, "test_loss": test_loss})

        if self.config.use_wandb:
            the_labels = ["t-shirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankleboot"]
            # print("logging confusion matrix from test set")
            
            predictio = np.argmax(y_pred, axis=1)
            wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
               probs=None, y_true=y, preds=predictio, class_names=the_labels
            )})
            wandb.finish()
        return test_loss, test_accuracy, predicted_labels
