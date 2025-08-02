import numpy as np
from activation_functions import *
import matplotlib.pyplot as plt

# 這裡是 DNN from scratch 的主程式與函數

## 參數初始化
def initialize_nn_parameters(layer_dims):
    parameters = {}
    L = len(layer_dims)  # number of layers
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters

def activation_list(ActList):
    act_list = []
    a, b = ActList[0], ActList[1]
    for l in range(1, a + b + 1):
        if l <= a:
            act_list.append('relu')
        else:
            act_list.append('sigmoid')
    return act_list

## 前向傳播
def forward_one(A_prev, W, b, activation):
    Z = np.dot(W, A_prev) + b
    if activation == "sigmoid":
        A = sigmoid(Z)
    elif activation == "relu":
        A = relu(Z)
    else:
        raise ValueError(f"Unsupported activation: {activation}")
    cache = (A, Z, A_prev, W, b, activation)
    return A, cache

def forward_L(X, parameters, act_list):
    caches = []
    A = X
    L = len(parameters) // 2
    for l in range(1, L + 1):
        A_prev = A
        activation = act_list[l - 1]
        A, cache = forward_one(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], activation)
        caches.append(cache)
    return A, caches

## 成本計算
def compute_cost(AL, Y, cost_function):
    if cost_function == "cross_entropy":
        cost = -1/(Y.shape[1]) * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
    cost = np.squeeze(cost)
    return cost

## 反向傳播
def backward_one(dA, cache):
    A, Z, A_prev, W, b, activation = cache
    m = A_prev.shape[1]
    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, A)
    elif activation == "relu":
        dZ = relu_backward(dA, Z)
    else:
        raise ValueError(f"Unsupported activation: {activation}")
    dW = 1/m * np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

def backward_L(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = caches[L - 1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = backward_one(dAL, current_cache)
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = backward_one(grads["dA" + str(l + 1)], current_cache)
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads

## 更新參數
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l+1)] -= learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] -= learning_rate * grads["db" + str(l+1)]
    return parameters

# 假設你已經有以下資料：
# x_train, y_train, x_test, y_test: numpy array, 維度需正確
# N: 輸入維度（例如 Ising 模型 N = L*L）

# train_and_evaluate DNN models
def train_and_evaluate(x_data, y_data, layers_dims, act_list, epochs=3000, learning_rate=0.01, verbose=True):
    """
    x_data: shape (N, m)
    y_data: shape (2, m)
    layers_dims: tuple, e.g. (N, 100, 2)
    act_list: list of activation functions per layer, e.g. ['relu', 'sigmoid']
    """
    # 切分 train/test
    m = x_data.shape[1]
    split = int(m * 0.8)
    x_train, x_test = x_data[:, :split], x_data[:, split:]
    y_train, y_test = y_data[:, :split], y_data[:, split:]

    parameters = initialize_nn_parameters(layers_dims)
    costs = []
    for epo in range(epochs):
        AL, caches = forward_L(x_train, parameters, act_list)
        cost = compute_cost(AL, y_train, cost_function="cross_entropy")
        grads = backward_L(AL, y_train, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
        if verbose and epo % 100 == 0:
            costs.append(cost)

    # 測試集評估
    AL_test, _ = forward_L(x_test, parameters, act_list)
    y_pred = np.argmax(AL_test, axis=0)
    y_true = np.argmax(y_test, axis=0)
    er_count = np.sum(y_pred != y_true)
    accuracy = 1.0 - float(er_count) / float(len(y_true))
    print("Number of prediction error is:", er_count)
    print("Accuracy of model is:", accuracy)

    # 學習曲線
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per 100)')
    plt.title(f"Learning rate = {learning_rate}")
    plt.show()

    return parameters, accuracy, costs