import numpy as np
from matplotlib import pyplot as plt


def Sigmoid(Z):
    Z = np.array(Z, dtype=np.float128)
    return 1 / (1 + np.exp(-Z))

def derivative_Sigmdoid(Z):
    f = 1 / (1 + np.exp(-Z))
    df = f * (1 - f)
    return df

def cost_function(o, nr_correct, l):
    # Cost / Error calculation
    e = 1 / len(o) * np.sum((o - l) ** 2, axis=0)
    nr_correct += int(np.argmax(o) == np.argmax(l))
    return e, nr_correct

def verify_classification(o, nr_correct, l):
    # Verify if is correct
    nr_correct += int(np.argmax(o) == np.argmax(l))
    return nr_correct
def forward_propagation(b_i_h, w_i_h, b_h_o, w_h_o,img):
    # Forward propagation input -> hidden
    h_pre = b_i_h + w_i_h @ img
    h = Sigmoid(h_pre)
    # Forward propagation hidden -> output
    o_pre = b_h_o + w_h_o @ h
    o = Sigmoid(o_pre)
    return o, h

def forward_propagation_one_image(b_i_h, w_i_h, b_h_o, w_h_o,img):
    img.shape += (1,)
    # Forward propagation input -> hidden
    h_pre = b_i_h + w_i_h @ img.reshape(784, 1)
    h = Sigmoid(h_pre)
    # Forward propagation hidden -> output
    o_pre = b_h_o + w_h_o @ h
    o = Sigmoid(o_pre)
    return o

def backward_propagation(b_i_h, w_i_h, b_h_o, w_h_o, o, l, h , img, learn_rate):
    # Backpropagation output -> hidden (cost function derivative)
    delta_o = o - l
    w_h_o += -learn_rate * delta_o @ np.transpose(h)
    b_h_o += -learn_rate * delta_o
    # Backpropagation hidden -> input (activation function derivative)
    delta_h = np.transpose(w_h_o) @ delta_o * (h * (1 - h))
    w_i_h += -learn_rate * delta_h @ np.transpose(img)
    b_i_h += -learn_rate * delta_h
    return b_i_h, w_i_h, b_h_o, w_h_o


def view_classification(train_images,b_i_h, w_i_h, b_h_o, w_h_o ):
    while True:
        index = int(input("Enter a number (0 - 59999): "))
        img = train_images[index]
        plt.imshow(img.reshape(28, 28), cmap="Greys")

        o = forward_propagation_one_image(b_i_h, w_i_h, b_h_o, w_h_o,img)

        plt.title(f"Final classification {o.argmax()}")
        plt.show()
    pass