import wandb
import argparse
import numpy as np
from keras.datasets import fashion_mnist, mnist

from neural_network import NeuralNetwork


def main(args):
    # wandb things
    if args.use_wandb.lower() == "true":
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
        wandb.run.name = (
            f"ac_{args.activation}_hl_{args.num_layers}_hs_{args.hidden_size}_bs_{args.batch_size}_"
            f"op_{args.optimizer}_ep_{args.epochs}"
        )

    # loading dataset from keras
    if args.dataset == "fashion_mnist":
        (train_img, train_labe), (test_img, test_labe) = fashion_mnist.load_data()
        the_labels = ["t-shirt", "trouser", "pullover", "dress", "coat","sandal", "shirt", "sneaker", "bag", "ankleboot"]

    elif args.dataset == "mnist":
        (train_img, train_labe), (test_img, test_labe) = mnist.load_data()
        the_labels = [str(i) for i in range(10)]
    else:
        raise ValueError("Invalid dataset (in lower case). Choose from ['mnist', 'fashion_mnist']")

    # train and valid split
    idx = np.random.permutation(train_img.shape[0])
    train_img, train_labe = train_img[idx], train_labe[idx]

    split = int(0.9 * len(train_img))
    val_img, val_labe = train_img[split:], train_labe[split:]
    train_img, train_labe = train_img[:split], train_labe[:split]

    # flattenning images , normalize to [0, 1]
    train_img = train_img.reshape(train_img.shape[0], -1) / 255.0
    val_img = val_img.reshape(val_img.shape[0], -1) / 255.0
    test_img = test_img.reshape(test_img.shape[0], -1) / 255.0

    print("Dataset splits:")
    print(f"Train set: {train_img.shape}, {train_labe.shape}")
    print(f"Valid set: {val_img.shape}, {val_labe.shape}")
    print(f"Test set: {test_img.shape}, {test_labe.shape}")

    # initialising neural network
    nn = NeuralNetwork(
        in_dim=784,
        out_dim=10,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        activation=args.activation,
        loss=args.loss,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        beta=args.beta,
        beta1=args.beta1,
        beta2=args.beta2,
        epsilon=args.epsilon,
        weight_decay=args.weight_decay,
        weight_init=args.weight_init,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_wandb=args.use_wandb.lower() == "true",
    )

    # train and validate
    nn.run(train_img, train_labe, val_img, val_labe)

    # testing the metwork
    y_pred = nn.test(test_img, test_labe)

    # confusion matrix to wandb
    if args.use_wandb.lower() == "true":
        wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=test_labe,
            preds=y_pred,
            class_names=the_labels
        )})
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a feedforward neural network")

    parser.add_argument("--use_wandb", type=str, default="false", help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="da6401_projectAss1", help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, default="username", help="WandB entity name (username)")
    parser.add_argument("--dataset", type=str, default="fashion_mnist", choices=["mnist", "fashion_mnist"], help="Dataset")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--loss", type=str, default="cross_entropy", choices=["mean_squared_error", "cross_entropy"], help="Loss function")
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], help="Optimizer")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.5, help="Momentum (for SGD variants)")
    parser.add_argument("--beta", type=float, default=0.5, help="Beta (for RMSProp)")
    parser.add_argument("--beta1", type=float, default=0.5, help="Beta1 (for Adam, Nadam)")
    parser.add_argument("--beta2", type=float, default=0.5, help="Beta2 (for Adam, Nadam)")
    parser.add_argument("--epsilon", type=float, default=0.000001, help="Epsilon (for optimizers)")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay (L2 regularization)")
    parser.add_argument("--weight_init", type=str, default="random", choices=["random", "Xavier"], help="Weight initialization method")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of hidden layers")
    parser.add_argument("--hidden_size", type=int, default=4, help="Number of neurons in each hidden layer")
    parser.add_argument("--activation", type=str, default="sigmoid", choices=["identity", "sigmoid", "tanh", "relu"], help="Activation function")

    args = parser.parse_args()
    main(args)
