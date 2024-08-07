import numpy as np
from typing import Literal


class Layer:
    def __init__(self, input_shape: tuple, n: int, name: str=None):
        assert isinstance(input_shape, tuple)
        assert isinstance(n, int)
        assert all(isinstance(n, int) for n in input_shape)
        self.input_shape = input_shape
        self.weights = np.array([], dtype=np.float32)
    def forward(self, X: np.ndarray) -> np.ndarray:
        assert X.shape == self.input_shape, f'input to layer {layer.name if layer.name else ""} has unexpected shape {X.shape} != {self.input_shape}'
        pass
    def backprop(self, outputs: np.ndarray, lr: float) -> np.ndarray:
        pass


class Dense(Layer):
    def __init__(self, input_shape: tuple, n: int):
        super().__init__(input_shape, n)
        assert len(input_shape) == 1, "input to dense layer must be a flat layer"
        self.weights = np.random.rand(*input_shape, n)
        self.bias = np.zeros(n)
    def forward(self, X: np.ndarray) -> np.ndarray:
        super().forward(X)
        return np.dot(X, self.weights) + self.bias
    def backprop(self, outputs: np.ndarray, lr: float) -> np.ndarray:
        grad = np.dot(outputs, self.weights.T)
        self.weights = self.weights - lr * grad
        return grad


class Activation(Layer):
    relu = np.vectorize(lambda x: max(0, x))
    relu_grad = np.vectorize(lambda y: int(y >= 0))
    def __init__(self, input_shape: tuple, n: int, activation_fn: Literal['relu', 'linear']):
        super().__init__(input_shape, n)
        self.activation_fn = activation_fn
    def forward(self, X: np.ndarray) -> np.ndarray:
        if self.activation_fn == 'linear':
            return X
        if self.activation_fn == 'relu':
            return Activation.relu(X)
    def backprop(self, outputs: np.ndarray, lr: float) -> np.ndarray:
        return Activation.relu_grad(outputs, lr)


if __name__ == '__main__':
    layer = Dense(input_shape=(3,), n=4)
    a = np.array([1, 2, 3])
    print(layer.weights)
    b = layer.forward(a)
    layer.backprop(b, 0.1)
    print(layer.weights)
