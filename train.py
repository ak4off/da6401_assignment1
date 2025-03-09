import wandb
import argparse
import numpy as np
from keras.datasets import fashion_mnist, mnist
from the_data import one_hot_encode
from neural_network import NeuralNetwork
from Q1_downloadnplot import download_and_plot

# def wandb_sweep():
def wandb_sweep(args, train_img, train_labe, val_img, val_labe):

    download_and_plot() # download and plot the images Question 1
    
    with wandb.init() as run:
        config = wandb.config
        epochs = config.epochs
        num_layers = config.num_layers
        learning_rate = config.learning_rate
        hidden_size = config.hidden_size
        weight_decay = config.weight_decay
        batch_size = config.batch_size
        optimizer = config.optimizer
        weight_init = config.weight_init
        activation = config.activation
        loss = config.loss

        run_name=f"ac_{activation}_hl_{num_layers}_hs_{hidden_size}_bs_{batch_size}_op_{optimizer}_ep_{epochs}"
        wandb.run.name=run_name
        nn = NeuralNetwork(
                use_wandb=args.use_wandb,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
                in_dim=784,
                out_dim=10,
                epochs=epochs,
                batch_size=batch_size,
                loss=loss,
                optimizer=optimizer,
                learning_rate=learning_rate,
                momentum=args.momentum,
                beta=args.beta,
                beta1=args.beta1,
                beta2=args.beta2,
                epsilon=args.epsilon,
                weight_decay=weight_decay,
                weight_init=weight_init,
                num_layers=num_layers,
                hidden_size=hidden_size,
                activation=activation,
            )
        nn.run(train_img, train_labe, val_img, val_labe)
        # test_loss, test_accuracy = nn.evaluate(test_img, test_labe)
        # wandb.log({"test_loss": test_loss, "test_accuracy": test_accuracy})
        # wandb.finish()

def main(args):

    sweep_config = {
        'method': 'bayes',  
        'name': 'sweep_loss_comparison',
        'metric': {'name': 'avg_valid_acc', 'goal': 'maximize'},
        'parameters': {
            'epochs': {'values': [5, 10,15, 100]},  
            'num_layers': {'values': [3, 4, 5]},
            'learning_rate': {'values': [1e-2, 1e-3, 1e-4]}, 
            'hidden_size': {'values': [32, 64, 128]},
            'weight_decay': {'values': [0, 0.0005, 0.5]},
            'batch_size': {'values': [16, 32, 64, 128, 256]},
            'optimizer': {'values': ['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam']},
            'weight_init': {'values': ['xavier', 'random']}, 
            'activation': {'values': ['sigmoid', 'tanh', 'relu']},
            'loss': {'values': ['cross_entropy', 'mean_squared_error']}  # Add squared error loss
        }
    }

    # sweep_config = {
    #     'method': 'bayes',  # Bayesian optimization for better hyperparameter search
    #     'name': 'sweep cross entropy',
    #     'metric': {'name': 'avg_valid_acc', 'goal': 'maximize'},
    #     'parameters': {
    #         'epochs': {'values': [5, 10, 15, 100]},  # Add 3 epochs to the list
    #         'num_layers': {'values': [3, 4, 5]},
    #         'learning_rate': {'values': [1e-2, 1e-3, 1e-4]},  # Include 0.01 (1e-2)
    #         'hidden_size': {'values': [128, 32, 64, 256]},  # Ensure 128 is included
    #         'weight_decay': {'values': [0, 0.0005, 0.005, 0.5]},
    #         'batch_size': {'values': [16, 32, 64, 128, 256]},
    #         'optimizer': {'values': ['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam']},
    #         'weight_init': {'values': ['xavier', 'random']},  # Ensure Xavier is included
    #         'activation': {'values': ['sigmoid', 'tanh', 'relu']},
    #         'loss': {'values': ['cross_entropy']}
    #     }
    # }

    # Load dataset
    if args.dataset == "fashion_mnist":
        (train_img, train_labe), (test_img, test_labe) = fashion_mnist.load_data()
        the_labels = ["t-shirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankleboot"]
    elif args.dataset == "mnist":
        (train_img, train_labe), (test_img, test_labe) = mnist.load_data()
        the_labels = [str(i) for i in range(10)]
    else:
        raise ValueError("Invalid dataset. Choose from ['mnist', 'fashion_mnist']")

    # Train-validation split
    idx = np.random.permutation(train_img.shape[0])
    train_img, train_labe = train_img[idx], train_labe[idx]
    split = int(0.9 * len(train_img))
    val_img, val_labe = train_img[split:], train_labe[split:]
    train_img, train_labe = train_img[:split], train_labe[:split]

    # Flatten and normalize
    train_img = train_img.reshape(train_img.shape[0], -1) / 255.0
    val_img = val_img.reshape(val_img.shape[0], -1) / 255.0
    test_img = test_img.reshape(test_img.shape[0], -1) / 255.0

    print("Dataset splits:")
    print(f"Train set: {train_img.shape}, {train_labe.shape}")
    print(f"Valid set: {val_img.shape}, {val_labe.shape}")
    print(f"Test set: {test_img.shape}, {test_labe.shape}")

    # WandB setup and sweep
    if args.use_wandb.lower() == "true":
        wandb.login()
        sweep_id = wandb.sweep(sweep=sweep_config, project=args.wandb_project)
        wandb.agent(sweep_id, function=lambda: wandb_sweep(args, train_img, train_labe, val_img, val_labe), count=10)
        wandb.finish()
        return  # If sweep is enabled, we do not train separately below

    # Initialize Neural Network
    nn = NeuralNetwork(
        in_dim=784, out_dim=10, num_layers=args.num_layers, hidden_size=args.hidden_size,
        activation=args.activation, loss=args.loss, optimizer=args.optimizer,
        learning_rate=args.learning_rate, momentum=args.momentum, beta=args.beta,
        beta1=args.beta1, beta2=args.beta2, epsilon=args.epsilon,
        weight_decay=args.weight_decay, weight_init=args.weight_init,
        epochs=args.epochs, batch_size=args.batch_size, use_wandb=args.use_wandb.lower() == "true"
    )

    # Train and validate
    nn.run(train_img, train_labe, val_img, val_labe)

    # Evaluate on test set
    test_loss, test_accuracy = nn.evaluate(test_img, test_labe)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    # Log test results if using WandB
    if args.use_wandb.lower() == "true":
        wandb.log({"test_loss": test_loss, "test_accuracy": test_accuracy})

    # Generate predictions for confusion matrix
    _, _, y_pred = nn.test(test_img, test_labe)

    # Log confusion matrix to WandB
    if args.use_wandb.lower() == "true":
        wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
            probs=None, y_true=test_labe, preds=y_pred, class_names=the_labels
        )})
        wandb.finish()
    # import wandb.sklearn 

    # wandb.sklearn.plot_confusion_matrix(test_labe, y_pred, labels=the_labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a feedforward neural network")

    parser.add_argument("-wdb","--use_wandb", type=str, default="false", help="Use Weights & Biases logging")
    parser.add_argument("-wp","--wandb_project", type=str, default="da6401_projectAss1", help="WandB project name")
    parser.add_argument("-we","--wandb_entity", type=str, default="username", help="WandB entity name (username)")
    parser.add_argument("-d","--dataset", type=str, default="fashion_mnist", choices=["mnist", "fashion_mnist"], help="Dataset")
    parser.add_argument("-e","--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("-b","--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("-l","--loss", type=str, default="cross_entropy", choices=["mean_squared_error", "cross_entropy"], help="Loss function")
    parser.add_argument("-o","--optimizer", type=str, default="sgd", choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], help="Optimizer")
    parser.add_argument("-lr","--learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("-m","--momentum", type=float, default=0.5, help="Momentum (for SGD variants)")
    parser.add_argument("-beta","--beta", type=float, default=0.5, help="Beta (for RMSProp)")
    parser.add_argument("-beta1","--beta1", type=float, default=0.5, help="Beta1 (for Adam, Nadam)")
    parser.add_argument("-beta2","--beta2", type=float, default=0.5, help="Beta2 (for Adam, Nadam)")
    parser.add_argument("-eps","--epsilon", type=float, default=0.000001, help="Epsilon (for optimizers)")
    parser.add_argument("-w_d","--weight_decay", type=float, default=0.0, help="Weight decay (L2 regularization)")
    parser.add_argument("-w_i","--weight_init", type=str, default="Xavier", choices=["random", "Xavier"], help="Weight initialization method")
    parser.add_argument("-nhl","--num_layers", type=int, default=3, help="Number of hidden layers")
    parser.add_argument("-sz","--hidden_size", type=int, default=128, help="Number of neurons in each hidden layer")
    parser.add_argument("-a","--activation", type=str, default="sigmoid", choices=["identity", "sigmoid", "tanh", "relu"], help="Activation function")


    parser.add_argument("-gdcw",'--gradient_clip_w', type=float, default=1.0, help='Max gradient clipping value for weights')
    parser.add_argument("-gdcb",'--gradient_clip_b', type=float, default=1.0, help='Max gradient clipping value for biases')

    args = parser.parse_args()
    main(args)
