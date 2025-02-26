import wandb
from keras.datasets import fashion_mnist
import numpy as np

wandb.init(
    project="ns24z274_da6401_assignment1",
    name="class_samples_q1"
)

(train_images, train_labels), _ = fashion_mnist.load_data()

class_desc = ["t-shirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankleboot"]


wandb_images = []
for class_id in range(10):
    for i in range(len(train_labels)):
        if train_labels[i] == class_id:
            img = train_images[i]
            wandb_images.append(wandb.Image(img, caption=f"{class_id}: {class_desc[class_id]}"))
            break

wandb.log({"Fashion_MNIST_Samples": wandb_images})

wandb.finish()
