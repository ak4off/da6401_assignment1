# **DA6401-Assignment1**

## **Overview**

This project builds a **feedforward neural network** from scratch to classify images from the Fashion-MNIST dataset. The model is trained using **gradient descent** and **backpropagation**, with **Weights & Biases (WandB)** for tracking experiments and visualizing results.

### **Assignment 1**: Feedforward Neural Network (Backpropagation from scratch)  
- [Assignment Link](https://wandb.ai/sivasankar1234/DA6401/reports/DA6401-Assignment-1--VmlldzoxMTQ2NDQwNw)  
- [Report Link] ()
- ![Image of question1](https://wandb.ai/ns24z274-iitm-ac-in/ns24z274_da6401_assignment1/runs/fky99vq0?nw=nwuserns24z274)

## **Setup**

### **Clone the repository:**  
```bash
   git clone https://github.com/ak4off/da6401_assignment1.git
   cd da6401_assignment1
```

### **Requirements**
```bash
pip install wandb
pip install numpy
pip install tensorflow  # Required for dataset download
```

## **Usage**

### **Arguments Explanation**

| Argument | Description | Default |
|----------|-------------|---------|
| `--use_wandb` | Use Weights & Biases logging (`true` or `false`) | `false` |
| `--wandb_project` | Name of the WandB project | `da6401_projectAss1` |
| `--wandb_entity` | WandB entity name (username) | `username` |
| `--dataset` | Dataset choice (`mnist` or `fashion_mnist`) | `fashion_mnist` |
| `--epochs` | Number of training epochs | `1` |
| `--batch_size` | Batch size for training | `4` |
| `--loss` | Loss function (`cross_entropy` or `mean_squared_error`) | `cross_entropy` |
| `--optimizer` | Optimizer choice (`sgd`, `momentum`, `adam`, etc.) | `sgd` |
| `--learning_rate` | Learning rate for optimization | `0.01` |
| `--num_layers` | Number of hidden layers | `3` |
| `--hidden_size` | Neurons per hidden layer | `128` |
| `--activation` | Activation function (`sigmoid`, `relu`, etc.) | `sigmoid` |
| `--weight_init` | Weight initialization (`random` or `Xavier`) | `Xavier` |

### **Example Commands**
```bash
python main.py --dataset fashion_mnist --epochs 10 --batch_size 32 --optimizer adam --learning_rate 0.001
```

## **Running the Model**
To train the feedforward neural network:
```bash
python main.py --dataset fashion_mnist --epochs 10 --batch_size 64 --optimizer adam
```

To enable WandB logging:
```bash
python main.py --use_wandb true --wandb_project my_project --wandb_entity my_username
```

## **Configuration & WandB Sweep**
The hyperparameter tuning is done using **WandB Sweeps**, which optimizes multiple parameters like:
- Number of layers
- Learning rate
- Optimizer selection
- Weight initialization
- Loss function choice

To run the WandB sweep:
```bash
wandb login
python main.py --use_wandb true
```

## **Dataset Handling**
- The dataset (`fashion_mnist` or `mnist`) is loaded using Keras.
- Images are normalized to [0,1] and reshaped to 784-dimensional vectors.
- A **90-10 train-validation split** is performed.

## **Neural Network Implementation**
The model is implemented in `neural_network.py`, using:
- **Forward Propagation**: Computes activations layer by layer.
- **Backpropagation**: Updates weights using gradients.
- **Optimizers**: Implements SGD, Adam, RMSProp, and others.

## **Results & Logging**
- Training loss and accuracy are recorded.
- **Test accuracy** is evaluated on the test set.
- **Confusion Matrix** is logged to WandB for analysis.

## **License**
This project is licensed under the MIT License.
