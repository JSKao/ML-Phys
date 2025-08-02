import numpy as np

def relu(Z):
    return np.maximum(0, Z)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def sigmoid(Z):
    return 1.0 / (1 + np.exp(-Z))

def sigmoid_backward(dA, A):
    return dA * A * (1 - A)