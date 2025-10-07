import numpy as np


def sigmoid(inputs: np.ndarray) -> np.ndarray:
    res = inputs.copy()
    pos = res > 0
    neg = res < 0
    res[pos] = 1.0 / (1 + np.exp(-res[pos]))
    res[neg] = np.exp(res[neg]) / (1 + np.exp(res[neg]))
    return res

def sigmoid_gradient(outputs: np.ndarray, out_gradients: np.ndarray) -> np.ndarray:
    return out_gradients * (outputs * (1 - outputs))

def softmax(inputs: np.ndarray) -> np.ndarray:
    exp_in = np.exp(inputs - np.max(inputs))
    sum_exp_out = np.reciprocal((exp_in.sum(axis=0)).astype(float))
    return exp_in * sum_exp_out

def softmax_gradient(outputs: np.ndarray, out_gradients: np.ndarray) -> np.ndarray:
    return out_gradients

def function(activation: str):
    match activation:
        case "sigmoid":
            return sigmoid
        case "softmax":
            return softmax
        case _:
            raise Exception("This activation function is does not exists or is not supported.")

def error(activation: str):
    match activation:
        case "sigmoid":
            return sigmoid_gradient
        case "softmax":
            return softmax_gradient
        case _:
            raise Exception("This activation function is does not exists or is not supported.")
