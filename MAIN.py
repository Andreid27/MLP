import pickle

import numpy as np

import NN_MNIST
from data import preprocess_data

# w = weights, b = bias, i = input, h = hidden, o = output, l = label
Sigmoid = "SIGMOID"
ReLu = "RELU"
nr_correct = 0

train_images, train_labels, test_images, test_labels = preprocess_data()

learn_rate = 0.01
epochs = 3
neurons_per_layer = [10, 8, 5, 2]
activation_function = Sigmoid
#TODO debug ReLu or replace ReLu function.

hidden_layers = len(neurons_per_layer) - 2
w_i, b_i = NN_MNIST.init_params(hidden_layers, neurons_per_layer)


for epochs in range(epochs):
    for img, l in zip(train_images, train_labels):
        print(type(train_images))
        print(type(train_labels))
        img.shape += (1,)
        l.shape += (1,)
        o, h = NN_MNIST.forward_propagation_layers(hidden_layers, b_i, w_i, img, activation_function)

        e, nr_correct = NN_MNIST.cost_function(o, nr_correct, l)

        b_i, w_i = NN_MNIST.backward_propagation_layers(hidden_layers, b_i, w_i, o, l, h, img, learn_rate, activation_function)
    # Show accuracy for this epoch
    print(f"Epoch: {epochs}")
    print(f"Acc for train set: {round((nr_correct / train_images.shape[0]) * 100, 2)}%")
    nr_correct = 0

    # Results for this epoch on test set
    for img, l in zip(test_images, test_labels):
        img.shape += (1,)
        l.shape += (1,)
        o, h = NN_MNIST.forward_propagation_layers(hidden_layers, b_i, w_i, img, activation_function)
        nr_correct = NN_MNIST.verify_classification(o, nr_correct, l)
    # Show accuracy for this epoch
    print(f"Acc for test set: {round((nr_correct / test_images.shape[0]) * 100, 2)}% \n \n")
    nr_correct = 0

with open("trained_params.pkl", "wb") as dump_file:
    pickle.dump((b_i, w_i), dump_file)
