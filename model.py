import numpy as np
from layers import Layer, Dense, Activation
from utils import Loss, MeanSquaredError


class Model:
    def __init__(self):
        self.layers: list[Layer] = []
    def add(self, layer: Layer) -> None:
        self.layers.append(layer)
    def compile(self, loss_fn: Loss) -> None:
        self.loss_fn = loss_fn
        for i, layer in enumerate(self.layers):
            if i == 0 and not layer.input_shape and not isinstance(layer, Activation):
                raise ValueError('input shape must be specified for the first layer in the model!')
            if i > 0:
                layer.input_shape = (self.layers[i - 1].n,)
            if isinstance(layer, Activation):
                layer.n = layer.input_shape
            layer.compile()
    def forward(self, X: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            X = layer.forward(X)
        return X
    def backprop(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        loss, grad = self.loss_fn(y_true=y_true, y_pred=y_pred)
        for layer in self.layers:
            grad = layer.backward(grad, 0.0001)
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        y_pred = self.forward(X=X)
        self.backprop(y_true=y, y_pred=y_pred)
    def summary(self) -> None:
        header = ('LayerName', 'Input Size', 'NumberParameters')
        summaries = [header] + [layer.summary() for layer in self.layers]
        # max width of the column
        m = max(max(len(str(elem)) for elem in summary) for summary in summaries) + 8
        sep = '\n' + '-' * m * 3 + '\n'
        result = ''
        for summary in summaries:
            result += sep
            result += '|'.join(' ' * ((m - len(str(elem)))//2) + str(elem) + ' ' * ((m - len(str(elem)))//2) for elem in summary)
        result += sep
        print(result)


if __name__ == '__main__':
    model = Model()
    model.add(Dense(input_shape=(1,), n=1, name='Dense2'))
    model.compile(loss_fn=MeanSquaredError())
    model.summary()
    print(model.layers[0].weights, model.layers[0].bias)
    x = np.array([[i] for i in range(100)])
    y = np.array([[i/2] for i in range(100)])
    model.fit(x, y)
    print(model.layers[0].weights, model.layers[0].bias)
