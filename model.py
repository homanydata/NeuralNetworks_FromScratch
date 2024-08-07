import numpy as np
from layers import Layer
from utils import Loss


class Model:
    def __init__(self):
        self.layers: list[Layer] = []
    def add(self, layer: Layer) -> None:
        if not isinstance(layer, Layer):
            raise TypeError('given layer is not of type Layer as expected.')
        self.layers.append(layer)
    def compile(self, loss_fn: Loss) -> None:
        self.loss_fn = loss_fn
    def forward(self, X: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            X = layer.forward(X)
        return X
    def backprop(self, outputs: np.ndarray) -> np.ndarray:
        loss, grad = self.loss_fn()
        for layer in self.layers:
            grad = layer.backprop(grad)
