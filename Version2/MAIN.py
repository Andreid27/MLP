import pickle

import NN_MNIST
from data import get_mnist
import numpy as np

# w = weights, b = bias, i = input, h = hidden, o = output, l = label

train_images, train_labels, _ , _ = get_mnist()
w_i_h = np.random.uniform(-0.5, 0.5, (256, 784))
w_h_o = np.random.uniform(-0.5, 0.5, (10, 256))
b_i_h = np.zeros((256, 1))
b_h_o = np.zeros((10, 1))

learn_rate = 0.01
nr_correct = 0
epochs = 3

for epochs in range(epochs):
    for img,l in zip(train_images, train_labels):
        img.shape += (1,)
        l.shape += (1,)
        o,h = NN_MNIST.forward_propagation(b_i_h, w_i_h, b_h_o, w_h_o,img)
        e, nr_correct = NN_MNIST.cost_function(o, nr_correct, l)
        b_i_h, w_i_h, b_h_o, w_h_o = NN_MNIST.backward_propagation(b_i_h, w_i_h, b_h_o, w_h_o, o, l, h, img, learn_rate)
    # Show accuracy for this epoch
    print(f"Acc: {round((nr_correct / train_images.shape[0]) * 100, 2)}%")
    nr_correct = 0

with open("trained_params.pkl","wb") as dump_file:
    pickle.dump((w_i_h, w_h_o, b_i_h, b_h_o),dump_file)