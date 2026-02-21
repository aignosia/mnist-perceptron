import json
import pickle
from pathlib import Path

import numpy as np

from .layer import Layer, LinearLayer
from .loss import Loss
from .utils import accuracy_score


class NN:
    def __init__(self, layers: list[Layer], loss_fn: Loss):
        self.layers = layers
        self.loss_fn = loss_fn
        self.results = {"lr": 0, "batch": 0, "loss_function": None, "results": []}

    def _forward(self, X: np.ndarray):
        input = X.copy()
        for layer in self.layers:
            input = layer.compute_outputs(input)

    def _backward(self, X: np.ndarray, y: np.ndarray, lr: float):
        grads = self.loss_fn.compute_grad(y, self.layers[-1].out)
        inputs = [X] + [layer.out for layer in self.layers[:-1]]
        for layer, input in zip(reversed(self.layers), reversed(inputs)):
            in_grads = layer.compute_gradients(grads)
            if isinstance(layer, LinearLayer):
                layer.adjust_weights(input, grads, lr)
            grads = in_grads

    def compute_loss(self, y: np.ndarray, y_pred: np.ndarray):
        return self.loss_fn.compute(y, y_pred)

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = X.copy()
        for layer in self.layers:
            out = layer.predict(out)
        return out

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int,
        batch: int,
        lr: float,
        X_test: np.ndarray,
        y_test: np.ndarray,
        result_path: str,
    ):
        if batch > X_train.shape[0]:
            raise Exception("batch size is bigger than training variables number.")
        self.results["lr"], self.results["batch"] = lr, batch
        self.results["loss_function"] = type(self.loss_fn).__name__
        for i in range(epochs):
            for j in range(0, X_train.shape[0], batch):
                s, e = j, min(j + batch, X_train.shape[0])
                self._forward(X_train[s:e])
                self._backward(X_train[s:e], y_train[s:e], lr)
            y_train_pred, y_test_pred = self.predict(X_train), self.predict(X_test)
            res = {"epoch": i + 1}
            res["train_loss"] = self.compute_loss(y_train, y_train_pred)
            res["train_acc"] = accuracy_score(y_train, y_train_pred)
            res["test_loss"] = self.compute_loss(y_test, y_test_pred)
            res["test_acc"] = accuracy_score(y_test, y_test_pred)
            print(
                f"epoch : {i + 1}"
                + f"\n  Train loss: {res['train_loss']:.4f} | Train acc.: {res['train_acc']:.4f}"
                + f"\n  Test loss:  {res['test_loss']:.4f} | Test acc.:  {res['test_acc']:.4f}"
            )
            self.results["results"].append(res)  # ty:ignore[unresolved-attribute]
        self.save(Path(result_path))

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        with open(path.joinpath("results.json"), "w") as f:
            json.dump(self.results, f, indent=2)
        with open(path.joinpath("layers.pkl"), "wb") as f:
            pickle.dump(self.layers, f)

    def load(self, path: Path):
        if not path.exists:
            raise FileNotFoundError(f"File {path} does not exist.")
        with open(path, "rb") as f:
            return pickle.load(f)
