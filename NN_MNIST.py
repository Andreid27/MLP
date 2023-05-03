import numpy as np
from matplotlib import pyplot as plt

def init_params(layers, neurons):
    w_i = []
    b_i = []
    for i in range(layers+1):
        w_i_h = np.random.uniform(-0.5, 0.5, (neurons[i+1], neurons[i]))
        b_i_h = np.zeros((neurons[i+1], 1))
        w_i.append(w_i_h)
        b_i.append(b_i_h)
    return w_i, b_i
def Sigmoid(Z):
    Z = np.array(Z, dtype=np.float128)
    return 1 / (1 + np.exp(-Z))

def derivative_Sigmdoid(Z):
    return Z*(1-Z)

def cost_function(o, nr_correct, l):
    # Cost / Error calculation
    e = 1 / len(o) * np.sum((o - l) ** 2, axis=0)
    nr_correct += int(np.argmax(o) == np.argmax(l))
    return e, nr_correct

def verify_classification(o, nr_correct, l):
    # Verify if is correct
    nr_correct += int(np.argmax(o) == np.argmax(l))
    return nr_correct
def forward_propagation(b_h_i, w_h_i,h):
    # Forward propagation hidden -> output
    o_pre = b_h_i + w_h_i @ h
    o = Sigmoid(o_pre)
    return o

def forward_propagation_one_image(b_i_h, w_i_h, b_h_o, w_h_o,img):
    img.shape += (1,)
    # Forward propagation input -> hidden
    h_pre = b_i_h + w_i_h @ img.reshape(784, 1)
    h = Sigmoid(h_pre)
    # Forward propagation hidden -> output
    o_pre = b_h_o + w_h_o @ h
    o = Sigmoid(o_pre)
    return o

def forward_propagation_layers(layers, b_i, w_i,img):
    h=[]
    o = forward_propagation(b_i[0], w_i[0], img)
    h.append(o)
    for i in range(layers):
        o = forward_propagation(b_i[i+1], w_i[i+1], o)
        h.append(o)
    return o,h

def backward_propagation(w_i, b_i, delta_o, h , i, learn_rate):
    # Backpropagation output -> hidden (cost function derivative)
    # delta_o  -> previous delta_h
    delta_h = np.transpose(w_i[i + 1]) @ delta_o * derivative_Sigmdoid(h[i + 1])
    w_i[i] += -learn_rate * delta_h @ np.transpose(h[i])
    b_i[i] += -learn_rate * delta_h
    return w_i, b_i, delta_h


def backward_propagation_layers(layers, b_i, w_i, o, l, h, img, learn_rate):
    # Backpropagation output -> hidden (cost function derivative)
    h.insert(0,img)
    delta_o = o - l
    w_i[layers] += -learn_rate * delta_o @ np.transpose(h[layers])
    b_i[layers] += -learn_rate * delta_o
    for i in range(layers-1, -1, -1):
        w_i, b_i, delta_o = backward_propagation(w_i,b_i,delta_o,h,i,learn_rate)
    return b_i, w_i


def view_classification(train_images, layers, b_i, w_i):
    while True:
        index = int(input("Enter a number (0 - 59999): "))
        img = train_images[index]
        plt.imshow(img.reshape(28, 28), cmap="Greys")

        o,_ = forward_propagation_layers(layers, b_i, w_i,img)

        plt.title(f"Final classification {o.argmax()}")
        plt.show()
    pass