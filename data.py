import numpy as np
import pathlib


def get_mnist():
    with np.load(f"{pathlib.Path(__file__).parent.absolute()}/data/mnist.npz") as data:
        train_images = data['x_train']
        train_labels = data['y_train']
        test_images = data['x_test']
        test_labels = data['y_test']
    train_images = train_images.astype("float32") / 255
    train_images = np.reshape(train_images, (train_images.shape[0], train_images.shape[1] * train_images.shape[2]))
    train_labels = np.eye(10)[train_labels]

    test_images = test_images.astype("float32") / 255
    test_images = np.reshape(test_images, (test_images.shape[0], test_images.shape[1] * test_images.shape[2]))
    test_labels = np.eye(10)[test_labels]

    return train_images, train_labels, test_images ,test_labels