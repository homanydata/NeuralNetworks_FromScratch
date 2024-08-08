import numpy as np
from typing import Literal


class Layer:
    def __init__(self, input_shape: tuple=None, name: str=None):
        self.name = name if name else self.__class__.__name__
        self.input_shape = input_shape
    def compile(self) -> None:
        self.weights = np.array([], dtype=np.float32)
    def forward(self, X: np.ndarray) -> np.ndarray:
        self.last_input = X
        pass
    def backward(self, outputs: np.ndarray, lr: float) -> np.ndarray:
        pass
    def summary(self) -> str:
        pass


class Dense(Layer):
    def __init__(self, n: int, input_shape: tuple=None, name: str=''):
        super().__init__(input_shape, n)
        self.name = name if name else self.__class__.__name__
        self.n = n
    def compile(self) -> None:
        super().compile()
        self.weights = np.random.rand(self.n, *self.input_shape)
        self.bias = np.zeros(shape=(self.n))
    def forward(self, X: np.ndarray) -> np.ndarray:
        super().forward(X)
        return np.dot(X, self.weights) + self.bias
    def backward(self, outputs: np.ndarray, lr: float) -> np.ndarray:
        w_grad = np.dot(self.last_input.T, outputs)
        b_grad = np.sum(outputs, axis=0)
        self.weights = self.weights - lr * w_grad
        self.bias = self.weights - lr * b_grad
        x_grad = np.dot(outputs, self.weights.T)
        return x_grad
    def summary(self) -> str:
        n_params = 1
        for d in self.weights.shape:
            n_params *= d
        n_params += self.bias.shape[0]
        return (self.name, self.input_shape, n_params)


class Activation(Layer):
    relu = np.vectorize(lambda x: max(0, x))
    relu_grad = np.vectorize(lambda y: int(y >= 0))
    def __init__(self, activation_fn: Literal['relu', 'linear']='linear', name: str=None):
        self.name = name if name else self.__class__.__name__
        self.activation_fn = activation_fn
    def forward(self, X: np.ndarray) -> np.ndarray:
        if self.activation_fn == 'linear':
            return X
        if self.activation_fn == 'relu':
            return Activation.relu(X)
    def backward(self, outputs: np.ndarray, lr: float) -> np.ndarray:
        return Activation.relu_grad(outputs, lr)
    def summary(self) -> tuple:
        return (self.name, self.input_shape, None)


if __name__ == '__main__':
    layer = Dense(input_shape=(3,), n=4)
    layer.compile()
    a = np.array([1, 2, 3])
    print(layer.weights)
    print(layer.weights.shape)
    b = layer.forward(a)
    print(b.shape)
