from typing import Protocol

import numpy as np


class Activation(Protocol):
    def function(self, inputs: np.ndarray) -> np.ndarray: ...
    def gradient(self, out: np.ndarray, out_grads: np.ndarray) -> np.ndarray: ...


class Sigmoid:
    def function(self, inputs: np.ndarray) -> np.ndarray:
        res = inputs.copy()
        res[res >= 0] = 1.0 / (1 + np.exp(-res[res >= 0]))
        res[res < 0] = np.exp(res[res < 0]) / (1 + np.exp(res[res < 0]))
        return res

    def gradient(self, out: np.ndarray, out_grads: np.ndarray) -> np.ndarray:
        return out_grads * out * (1 - out)


class Softmax:
    def function(self, inputs: np.ndarray) -> np.ndarray:
        max = np.array([np.max(inputs, axis=1)]).T
        diff = inputs - max
        sum = np.array([np.exp(diff).sum(axis=1)]).T
        return diff - np.log(sum)

    def gradient(self, out: np.ndarray, out_grads: np.ndarray) -> np.ndarray:
        return out_grads
