import pickle

import NN_MNIST
from data import get_mnist

_, _, test_images , test_labels = get_mnist()

with open("trained_params.pkl","rb") as dump_file:
    w_i_h, w_h_o, b_i_h, b_h_o = pickle.load(dump_file)

nr_correct = 0

for img, l in zip(test_images, test_labels):
    img.shape += (1,)
    l.shape += (1,)
    o, h = NN_MNIST.forward_propagation(b_i_h, w_i_h, b_h_o, w_h_o, img)
    nr_correct = NN_MNIST.verify_classification(o, nr_correct, l)

print(f"Acc: {round((nr_correct / test_images.shape[0]) * 100, 2)}%")

NN_MNIST.view_classification(test_images,b_i_h, w_i_h, b_h_o, w_h_o )