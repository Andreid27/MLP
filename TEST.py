import pickle
from matplotlib import pyplot as plt

import NN_MNIST
from data import get_mnist

_, _, test_images , test_labels = get_mnist()
Sigmoid = "SIGMOID"
ReLu = "RELU"

with open("trained_params.pkl","rb") as dump_file:
    b_i, w_i = pickle.load(dump_file)

learn_rate = 0.01
nr_correct = 0
neurons_per_layer = [784, 40, 20, 10]
hidden_layers = len(neurons_per_layer) - 2


nr_correct = 0

for img, l in zip(test_images, test_labels):
    img.shape += (1,)
    l.shape += (1,)
    o, h = NN_MNIST.forward_propagation_layers(hidden_layers, b_i, w_i, img, ReLu)
    nr_correct = NN_MNIST.verify_classification(o, nr_correct, l)

print(f"Acc: {round((nr_correct / test_images.shape[0]) * 100, 2)}%")

# NN_MNIST.view_classification(test_images, hidden_layers, b_i, w_i)