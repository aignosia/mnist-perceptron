from typing import Protocol

import numpy as np


class Loss(Protocol):
    def compute(self, y: np.ndarray, y_pred: np.ndarray) -> np.ndarray: ...
    def compute_grad(self, y: np.ndarray, y_pred: np.ndarray) -> np.ndarray: ...


class CrossEntropyLoss:
    def _log_softmax(self, logits: np.ndarray):
        diff = logits - logits.max(axis=1, keepdims=True)
        return diff - np.log(np.exp(diff).sum(axis=1, keepdims=True))

    def compute(self, y: np.ndarray, y_pred: np.ndarray):
        return np.sum(-(y * self._log_softmax(y_pred))).item() / y.shape[0]

    def compute_grad(self, y: np.ndarray, y_pred: np.ndarray):
        return np.exp(self._log_softmax(y_pred)) - y


class L2Loss:
    def compute(self, y: np.ndarray, y_pred: np.ndarray):
        return np.sum(np.power(y_pred - y, 2)).item() / y.shape[0]

    def compute_grad(self, y: np.ndarray, y_pred: np.ndarray):
        return y_pred - y
