import numpy as np

class Trainer:
    def __init__(self, layers, learning_rate=0.01):
        self.layers = layers
        self.learning_rate = learning_rate
        self.costs = []
        self.metrics = {}

    def forward(self, X):
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def backward(self, dAL):
        dA = dAL
        grads = []
        for layer in reversed(self.layers):
            dA, dW, db = layer.backward(dA)
            grads.insert(0, (dW, db))
        return grads

    def update(self, grads):
        for i, layer in enumerate(self.layers):
            dW, db = grads[i]
            layer.W -= self.learning_rate * dW
            layer.b -= self.learning_rate * db

    def compute_cost(self, AL, Y):
        # binary/multiclass cross-entropy
        cost = -np.mean(Y * np.log(AL + 1e-8) + (1 - Y) * np.log(1 - AL + 1e-8))
        return cost

    def fit(self, X, Y, epochs=1000, verbose=100, X_val=None, Y_val=None):
        m = Y.shape[1]
        for epoch in range(epochs):
            AL = self.forward(X)
            cost = self.compute_cost(AL, Y)
            if verbose and epoch % verbose == 0:
                self.costs.append(cost)
            dAL = -(Y / (AL + 1e-8) - (1 - Y) / (1 - AL + 1e-8)) / m
            grads = self.backward(dAL)
            self.update(grads)
        return self

    def predict(self, X):
        AL = self.forward(X)
        return np.argmax(AL, axis=0)

    def score(self, X, Y):
        y_pred = self.predict(X)
        y_true = np.argmax(Y, axis=0)
        return np.mean(y_pred == y_true)
