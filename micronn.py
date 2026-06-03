import copy
import json
import pickle
from functools import singledispatch
from pathlib import Path
from typing import Callable

import numpy as np


class Linear:
    def __init__(self, input_size: int, size: int):
        self.input_size = input_size
        self.size = size
        self.B = np.random.normal(0, 1, size=(size, 1))
        self.W = np.random.normal(0, 1, size=(size, input_size))
        self.out = np.empty(0)
        self.grads = np.empty(0)


class Activation:
    def __init__(self, func: Callable, grad_func: Callable):
        self.func = func
        self.grad_func = grad_func
        self.out = np.empty(0)
        self.grads = np.empty(0)


@singledispatch
def compute_layer_output(arg):
    return arg


@compute_layer_output.register
def _(layer: Linear, X: np.ndarray) -> np.ndarray:
    B_W = np.concatenate((layer.B, layer.W), axis=1)
    X_prime = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1).T
    return np.matmul(B_W, X_prime).T


@compute_layer_output.register
def _(layer: Activation, X: np.ndarray) -> np.ndarray:
    return layer.func(X)


@singledispatch
def compute_layer_gradient(arg):
    return arg


@compute_layer_gradient.register
def _(layer: Linear, out_grads: np.ndarray) -> np.ndarray:
    return np.matmul(out_grads, layer.W)


@compute_layer_gradient.register
def _(layer: Activation, out_grads: np.ndarray) -> np.ndarray:
    return layer.grad_func(layer, out_grads)


def adjust_weights(
    layer: Linear, X: np.ndarray, grads: np.ndarray, lr: float
) -> tuple[np.ndarray, np.ndarray]:
    W_grads = np.matmul(X.T, grads) / X.shape[0]
    B_grads = np.matmul(np.ones((1, X.shape[0])), grads) / X.shape[0]
    W, B = layer.W - lr * W_grads.T, layer.B - lr * B_grads.T
    return W, B


class Loss:
    def __init__(self, name: str, loss: np.ndarray, grad: np.ndarray):
        self.name = name
        self.loss = loss
        self.grads = grad


class NN:
    def __init__(self, *layers: Linear | Activation):
        self.layers = list(layers)


def softmax(X: np.ndarray):
    exp = np.exp(X - X.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def cross_entropy_loss(y: np.ndarray, y_logits: np.ndarray) -> Loss:
    diff = y_logits - y_logits.max(axis=1, keepdims=True)
    log_softmax = diff - np.log(np.exp(diff).sum(axis=1, keepdims=True))
    loss = np.sum(-(y * log_softmax)).item() / y.shape[0]
    return Loss("Cross-Entropy", loss, np.exp(log_softmax) - y)


def forward(model: NN, X: np.ndarray) -> NN:
    out_model, out = copy.deepcopy(model), X
    for layer in out_model.layers:
        layer.out = compute_layer_output(layer, out)
        out = layer.out
    return out_model


def backward(model: NN, X: np.ndarray, loss: Loss, lr: float) -> NN:
    out_model, grads = copy.deepcopy(model), loss.grads
    reversed_inputs = [e.out for e in reversed(out_model.layers[:-1])] + [X]
    for layer, input in zip(reversed(out_model.layers), reversed_inputs):
        layer.grads = compute_layer_gradient(layer, grads)
        if isinstance(layer, Linear):
            layer.W, layer.B = adjust_weights(layer, input, grads, lr)
        grads = layer.grads
    return out_model


def one_hot_encode(digit: int) -> list:
    if not 0 <= digit <= 9:
        raise Exception("Give a digit between 0 and 9.")
    return [0] * digit + [1] + [0] * (9 - digit)


def accuracy_score(y: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.argmax(y, axis=1) == np.argmax(y_pred, axis=1)) * 100


def manual_seed(seed):
    np.random.seed(seed)


print("Preprocess data...")
data = np.load("data/mnist.npz")
X_train = data["x_train"].reshape(data["x_train"].shape[0], -1)
y_train = np.array([np.array(one_hot_encode(y)) for y in data["y_train"]])
X_test = data["x_test"].reshape(data["x_test"].shape[0], -1)
y_test = np.array([np.array(one_hot_encode(y)) for y in data["y_test"]])
print("Setting model...")
manual_seed(42)
model = NN(Linear(784, 10))
loss_fn, lr, epochs, batch = cross_entropy_loss, 1e-2, 5, 128
results = {"lr": lr, "batch": batch, "loss_function": "", "results": []}
print("Training model...")
for epoch in range(epochs):
    for i in range(0, X_train.shape[0], batch):
        e = min(i + batch, X_train.shape[0])
        X_batch, y_batch = X_train[i:e], y_train[i:e]
        model = forward(model, X_batch)
        batch_loss = loss_fn(y_batch, model.layers[-1].out)
        model = backward(model, X_batch, batch_loss, lr)
    results["loss_function"], res = batch_loss.name, {}
    y_logits = forward(model, X_train).layers[-1].out
    y_test_logits = forward(model, X_test).layers[-1].out
    y_pred, y_test_pred = softmax(y_logits), softmax(y_test_logits)
    loss, acc = loss_fn(y_train, y_logits), accuracy_score(y_train, y_pred)
    test_loss = loss_fn(y_test, y_test_logits)
    test_acc = accuracy_score(y_test, y_test_pred)
    print_res = f"Epoch: {epoch + 1}\n  Loss: {loss.loss:.4f} | Acc.: {acc:.4f}\n  Test loss:  {test_loss.loss:.4f} | Test acc.:  {test_acc:.4f}"
    print(print_res)
    res["epoch"], res["loss"], res["acc"] = epoch + 1, loss.loss, acc
    res["test_loss"], res["test_acc"] = test_loss.loss, test_acc
    results["results"].append(res)
print("Saving results...")
base_path = Path("models/micronn_ex")
base_path.mkdir(parents=True, exist_ok=True)
pickle.dump(model, open(base_path.joinpath("model.pkl"), "wb"))
json.dump(results, open(base_path.joinpath("results.json"), "w"), indent=2)
