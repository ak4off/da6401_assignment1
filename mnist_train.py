import wandb
import argparse
import numpy as np
from keras.datasets import fashion_mnist, mnist
from the_data import one_hot_encode
from neural_network import NeuralNetwork
from Q1_downloadnplot import download_and_plot


def wandb_sweep(args, train_img, train_labe, val_img, val_labe):
    """ Runs a WandB sweep using the predefined hyperparameter configurations. """
    download_and_plot()  # Download and visualize images (Q1)

    with wandb.init() as run:
        config = wandb.config

        # Set meaningful run names
        run_name = (
            f"ac_{config.activation}_hl_{config.num_layers}_hs_{config.hidden_size}"
            f"_bs_{config.batch_size}_op_{config.optimizer}_ep_{config.epochs}_lr_{config.learning_rate}"
        )
        wandb.run.name = run_name

        # Initialize neural network with WandB-configured parameters
        nn = NeuralNetwork(
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            in_dim=784,
            out_dim=10,
            epochs=config.epochs,
            batch_size=config.batch_size,
            loss=config.loss,
            optimizer=config.optimizer,
            learning_rate=config.learning_rate,
            momentum=args.momentum,
            beta=args.beta,
            beta1=args.beta1,
            beta2=args.beta2,
            epsilon=args.epsilon,
            weight_decay=config.weight_decay,
            weight_init=config.weight_init,
            num_layers=config.num_layers,
            hidden_size=config.hidden_size,
            activation=config.activation,
        )

        # Run training and validation
        nn.run(train_img, train_labe, val_img, val_labe)


def main(args):

    # Define the 4 fixed hyperparameter configurations
    sweep_config = {
        "method": "bayes",
        "name": "mnist_best_configs",
        "metric": {"name": "avg_valid_acc", "goal": "maximize"},
        "parameters": {
            "dataset": {"values": ["mnist"]},
            "activation": {"values": ["sigmoid", "relu","tanh"]},
            "batch_size": {"values": [4, 64, 128, 256]},  
            "epochs": {"values": [20,30]},
            "hidden_size": {"values": [128, 120, 100, 128]},
            "learning_rate": {"values": [0.01, 0.003, 0.005, 0.003]},
            "loss": {"values": ["cross_entropy"]},
            "num_layers": {"values": [3, 4, 5]},
            "optimizer": {"values": ["adam"]},
            "weight_decay": {"values": [0, 0.001, 0.01, 0.005]},
            "weight_init": {"values": ["xavier"]}
        }
    }

    # Load dataset
    if args.dataset == "fashion_mnist":
        (train_img, train_labe), (test_img, test_labe) = fashion_mnist.load_data()
    elif args.dataset == "mnist":
        (train_img, train_labe), (test_img, test_labe) = mnist.load_data()
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

    # Run WandB sweep if enabled
    if args.use_wandb.lower() == "true":
        wandb.login()
        sweep_id = wandb.sweep(sweep=sweep_config, project=args.wandb_project)
        wandb.agent(sweep_id, function=lambda: wandb_sweep(args, train_img, train_labe, val_img, val_labe), count=4)
        wandb.finish()
        return  # Exit after sweeps

    # Standard Training (if WandB is not enabled)
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

    # Log test results to WandB
    if args.use_wandb.lower() == "true":
        wandb.log({"test_loss": test_loss, "test_accuracy": test_accuracy})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a feedforward neural network")

    # WandB parameters
    parser.add_argument("-wdb", "--use_wandb", type=str, default="false", help="Use Weights & Biases logging")
    parser.add_argument("-wp", "--wandb_project", type=str, default="da6401_projectAss1", help="WandB project name")
    parser.add_argument("-we", "--wandb_entity", type=str, default="username", help="WandB entity name (username)")

    # Dataset selection
    parser.add_argument("-d", "--dataset", type=str, default="fashion_mnist", choices=["mnist", "fashion_mnist"], help="Dataset")

    # Hyperparameters
    parser.add_argument("-e", "--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("-l", "--loss", type=str, default="cross_entropy", choices=["mean_squared_error", "cross_entropy"], help="Loss function")
    parser.add_argument("-o", "--optimizer", type=str, default="sgd", choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], help="Optimizer")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("-m", "--momentum", type=float, default=0.5, help="Momentum (for SGD variants)")
    parser.add_argument("-beta", "--beta", type=float, default=0.5, help="Beta (for RMSProp)")
    parser.add_argument("-beta1", "--beta1", type=float, default=0.5, help="Beta1 (for Adam, Nadam)")
    parser.add_argument("-beta2", "--beta2", type=float, default=0.5, help="Beta2 (for Adam, Nadam)")
    parser.add_argument("-eps", "--epsilon", type=float, default=0.000001, help="Epsilon (for optimizers)")
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0, help="Weight decay (L2 regularization)")
    parser.add_argument("-w_i", "--weight_init", type=str, default="Xavier", choices=["random", "Xavier"], help="Weight initialization method")
    parser.add_argument("-nhl", "--num_layers", type=int, default=3, help="Number of hidden layers")
    parser.add_argument("-sz", "--hidden_size", type=int, default=128, help="Number of neurons in each hidden layer")

    args = parser.parse_args()
    main(args)
