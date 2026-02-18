from typing import Protocol

import numpy as np


class Loss(Protocol):
    def function(self, y: np.ndarray, y_pred: np.ndarray) -> np.ndarray: ...
    def gradient(self, y: np.ndarray, y_pred: np.ndarray) -> np.ndarray: ...


class CrossEntropy:
    def function(self, y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.sum(-(y * y_pred).sum(axis=1)) / y.shape[0]

    def gradient(self, y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return y_pred - y


class L2Loss:
    def function(self, y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.sum(np.power(y_pred - y, 2)) / y.shape[0]

    def gradient(self, y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return y_pred - y


if __name__ == "__main__":
    y = np.array([[0, 0, 1]])
    y_pred = np.array([[0.2, 0.3, 0.5]])
    normal_ce = -np.sum(y * np.log(y_pred)) / y.shape[0]
    print(normal_ce)
    print(CrossEntropy().function(y, y_pred))
