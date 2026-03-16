import copy
import json
import pickle
from pathlib import Path
from typing import Callable

import numpy as np


class Linear:
    def __init__(self, input_size: int, size: int):
        if size <= 0 or input_size <= 0:
            raise Exception("Layer size or input size can't be negative or null.")
        self.W = np.random.normal(0.0, 1.0, (size, input_size))
        self.B = np.random.normal(0.0, 1.0, (size, 1))


class Activation:
    def __init__(self, func: Callable, grad: Callable):
        self.func = func
        self.grad = grad


class Layer:
    out: np.ndarray

    def __init__(self, linear: Linear, activation: Activation | None = None):
        self.linear = linear
        self.activation = activation


class Loss:
    def __init__(self, name: str, loss: np.ndarray, grad: np.ndarray):
        self.name = name
        self.loss = loss
        self.grad = grad


class NN:
    def __init__(self, *layers: Layer):
        self.layers = list(layers)


def softmax(X: np.ndarray):
    exp = np.exp(X - X.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def cross_entropy_loss(y_logits: np.ndarray, y_pred: np.ndarray) -> Loss:
    diff = y_pred - y_pred.max(axis=1, keepdims=True)
    log_softmax = diff - np.log(np.exp(diff).sum(axis=1, keepdims=True))
    loss = np.sum(-(y_logits * log_softmax)).item() / y_logits.shape[0]
    return Loss("Cross-Entropy", loss, np.exp(log_softmax) - y_logits)


def forward(nn: NN, X: np.ndarray) -> NN:
    out_nn, out = copy.deepcopy(nn), X.copy()
    for layer in out_nn.layers:
        B_W = np.concatenate((layer.linear.B, layer.linear.W), axis=1)
        X_prime = np.concatenate((np.ones((out.shape[0], 1)), out), axis=1).T
        out = np.matmul(B_W, X_prime).T
        out = layer.activation.func(out) if layer.activation else out
        layer.out = out
    return out_nn


def backward(nn: NN, X: np.ndarray, y: np.ndarray, loss: Loss, lr: float) -> NN:
    out_nn, grads = copy.deepcopy(nn), loss.grad
    _X = [X] + [e.out for e in out_nn.layers[:-1]]
    for layer, input in zip(reversed(out_nn.layers), reversed(_X)):
        grads = layer.activation.grad(layer, grads) if layer.activation else grads
        in_grads = np.matmul(grads, layer.linear.W)
        W_grads = np.matmul(input.T, grads) / X.shape[0]
        B_grads = np.matmul(np.ones((1, input.shape[0])), grads) / input.shape[0]
        layer.linear.W = layer.linear.W - lr * W_grads.T
        layer.linear.B = layer.linear.B - lr * B_grads.T
        grads = in_grads
    return out_nn


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
model = NN(Layer(Linear(784, 10)))
loss_fn, lr, epochs, batch = cross_entropy_loss, 1e-2, 5, 128
results, res = {"lr": lr, "batch": batch, "loss_function": "", "results": []}, {}
print("Training model...")
for epoch in range(epochs):
    for i in range(0, X_train.shape[0], batch):
        e = min(i + batch, X_train.shape[0])
        X_batch, y_batch = X_train[i:e], y_train[i:e]
        model = forward(model, X_batch)
        batch_loss = loss_fn(y_batch, forward(model, X_batch).layers[-1].out)
        model = backward(model, X_batch, y_batch, batch_loss, lr)
    results["loss_function"] = batch_loss.name
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
    results["results"].append(res)  # ty:ignore[unresolved-attribute]
print("Saving results...")
base_path = Path("models/micronn")
base_path.mkdir(parents=True, exist_ok=True)
pickle.dump(model, open(base_path.joinpath("model.pkl"), "wb"))
json.dump(results, open(base_path.joinpath("results.json"), "w"))
