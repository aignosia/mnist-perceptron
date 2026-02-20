from __future__ import annotations

from typing import Protocol

import numpy as np


class Layer(Protocol):
    out: np.ndarray

    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def compute_outputs(self, X: np.ndarray) -> np.ndarray: ...
    def compute_gradients(self, out_grads: np.ndarray) -> np.ndarray: ...


class LinearLayer:
    def __init__(self, size: int, input_size: int):
        self.size = size
        self.out = np.array([])
        self.input_size = input_size
        self.W = np.random.normal(0.0, 1.0, (self.size, self.input_size))
        self.B = np.random.normal(0.0, 1.0, (self.size, 1))
        self.grads = np.array([])

    def predict(self, X: np.ndarray):
        B_W = np.concatenate((self.B, self.W), axis=1)
        X_prime = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1).T
        return np.matmul(B_W, X_prime).T

    def compute_outputs(self, X: np.ndarray):
        self.out = self.predict(X)
        return self.out

    def compute_gradients(self, out_grads: np.ndarray):
        self.grads = np.matmul(out_grads, self.W)
        return self.grads

    def adjust_weights(self, X: np.ndarray, out_grads: np.ndarray, lr: float):
        W_grads = np.matmul(X.T, out_grads) / X.shape[0]
        B_grads = np.matmul(np.ones((1, X.shape[0])), out_grads) / X.shape[0]
        self.W = self.W - lr * W_grads.T
        self.B = self.B - lr * B_grads.T


class SigmoidLayer:
    def predict(self, X: np.ndarray):
        res = X.copy()
        res[res >= 0] = 1.0 / (1 + np.exp(-res[res >= 0]))
        res[res < 0] = np.exp(res[res < 0]) / (1 + np.exp(res[res < 0]))
        return res

    def compute_outputs(self, X: np.ndarray):
        self.out = self.predict(X)
        return self.out

    def compute_gradients(self, out_grads: np.ndarray):
        return out_grads * self.out * (1 - self.out)


class SoftmaxLayer:
    def predict(self, X: np.ndarray):
        exp = np.exp(X - X.max(axis=1, keepdims=True))
        sum = exp.sum(axis=1, keepdims=True)
        return exp / sum

    def compute_outputs(self, X: np.ndarray):
        self.out = self.predict(X)

    def compute_gradients(self, out_grads: np.ndarray):
        return out_grads
