
import numpy as np
from abc import ABC, abstractmethod

class BaseLayer(ABC):
    """抽象層，所有 layer 都要繼承"""
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, X):
        pass

    @abstractmethod
    def backward(self, dA):
        pass

class DenseLayer(BaseLayer):
    def __init__(self, in_dim, out_dim, activation):
        super().__init__()
        self.W = np.random.randn(out_dim, in_dim) * 0.01
        self.b = np.zeros((out_dim, 1))
        self.activation = activation
        self.cache = None

    def forward(self, X):
        Z = np.dot(self.W, X) + self.b
        if self.activation == 'relu':
            A = np.maximum(0, Z)
        elif self.activation == 'sigmoid':
            A = 1 / (1 + np.exp(-Z))
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")
        self.cache = (X, Z, A)
        return A

    def backward(self, dA):
        X, Z, A = self.cache
        m = X.shape[1]
        if self.activation == 'sigmoid':
            dZ = dA * A * (1 - A)
        elif self.activation == 'relu':
            dZ = dA * (Z > 0)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")
        dW = 1/m * np.dot(dZ, X.T)
        db = 1/m * np.sum(dZ, axis=1, keepdims=True)
        dX = np.dot(self.W.T, dZ)
        return dX, dW, db

def build_layers_from_config(layers_config):
    """
    根據 config 建立 layer list。每個 config dict 需有 'type' 欄位。
    支援 DenseLayer，未來可擴充 CNN/GCN/Transformer/VAE。
    """
    layers = []
    for cfg in layers_config:
        ltype = cfg['type'].lower()
        if ltype == 'dense':
            layers.append(DenseLayer(cfg['in_dim'], cfg['out_dim'], cfg['activation']))
        elif ltype == 'conv':
            layers.append(ConvLayer())  # TODO: 傳入參數
        elif ltype == 'gcn':
            layers.append(GCNLayer())   # TODO: 傳入參數
        elif ltype == 'transformer':
            layers.append(TransformerLayer())
        elif ltype == 'vae':
            layers.append(VAELayer())
        else:
            raise ValueError(f"Unknown layer type: {cfg['type']}")
    return layers

# 預留 CNN/GCN/Transformer/VAE layer class
class ConvLayer(BaseLayer):
    def __init__(self):
        super().__init__()
        # TODO: add parameters and implement
    def forward(self, X):
        pass
    def backward(self, dA):
        pass

class GCNLayer(BaseLayer):
    def __init__(self):
        super().__init__()
        # TODO: add parameters and implement
    def forward(self, X, A):
        pass
    def backward(self, dA):
        pass

class TransformerLayer(BaseLayer):
    def __init__(self):
        super().__init__()
        # TODO: add parameters and implement
    def forward(self, X):
        pass
    def backward(self, dA):
        pass

class VAELayer(BaseLayer):
    def __init__(self):
        super().__init__()
        # TODO: add parameters and implement
    def forward(self, X):
        pass
    def backward(self, dA):
        pass
