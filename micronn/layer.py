from __future__ import annotations

import numpy as np

from .activations import Activation, Softmax


class Layer:
    def __init__(
        self, size: int, activation: Activation, input_size: int | None = None
    ):
        self.size = size
        self.out = np.array([])
        self.activation = activation
        self.input_size = input_size
        self.grads = {"W": np.array([]), "B": np.array([])}

    def initialize_layer(self):
        if self.input_size is None:
            raise Exception("Input size should be superior than 0.")
        self.W = np.random.normal(0.0, 1.0, (self.size, self.input_size))
        self.B = np.random.normal(0.0, 1.0, (self.size, 1))

    def _compute_outputs(self, X: np.ndarray):
        B_W = np.concatenate((self.B, self.W), axis=1)
        X_prime = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1).T
        out = (B_W @ X_prime).T
        return self.activation.function(out)  # Transpose to make like y

    def compute_outputs(self, X: np.ndarray):
        self.out = self._compute_outputs(X)
        if isinstance(self.activation, Softmax):
            self.log_out = self._compute_outputs(X)
            self.out = np.exp(self._compute_outputs(X))

    def predict(self, X: np.ndarray):
        if isinstance(self.activation, Softmax):
            return self._compute_outputs(X)
        return self._compute_outputs(X)

    def compute_gradients(
        self, inputs: np.ndarray, out_grads: np.ndarray, batch: int
    ):
        activation_grads = self.activation.gradient(self.out, out_grads)
        self.grads["W"] = (inputs.T @ activation_grads) / batch
        self.grads["B"] = (np.ones((1, batch)) @ activation_grads) / batch

    def adjust_weights(self, lr: float):
        if self.grads["W"].size and self.grads["B"].size == 0:
            raise Exception("Compute the gradients first.")
        self.W = self.W - lr * self.grads["W"].T
        self.B = self.B - lr * self.grads["B"].T
