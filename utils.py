import numpy as np


class Loss:
    def __init__(self):
        pass
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
        """
        Returns loss (float) and loss gradient (np.ndarray)
        """
        if not y_pred.shape == y_true.shape:
            raise ValueError(f'y_pred has unexpected shape {y_pred.shape} != {y_true.shape}')
        pass


class MeanSquaredError(Loss):
    def __init__(self):
        super().__init__()
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
        super().__call__(y_true, y_pred)
        return np.sum((y_true - y_pred) ** 2), -2 * (y_true - y_pred)


if __name__ == '__main__':
    loss = MeanSquaredError()
    a = np.array([1, 2, 3, 4])
    b = np.array([1, 0, 2, 5])
    print(loss(a, b))
