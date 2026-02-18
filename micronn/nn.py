import json
import pickle
from pathlib import Path

import numpy as np

from micronn.activations import Softmax

from .layer import Layer
from .loss import CrossEntropy, Loss
from .utils import accuracy_score


class NN:
    def __init__(self, layers: list[Layer], loss_func: Loss):
        if layers[0].input_size is None:
            raise Exception("You must provide input size for the first layer.")
        if isinstance(loss_func, CrossEntropy) and not isinstance(
            layers[-1].activation, Softmax
        ):
            raise Exception("Only softmax is supported for cross-entropy loss.")
        self.layers = layers
        for i in range(len(self.layers)):
            if i > 0:
                self.layers[i].input_size = self.layers[i - 1].size
            self.layers[i].initialize_layer()
        self.loss_func = loss_func
        self.results = []

    def _forward(self, X: np.ndarray):
        input = X.copy()
        for layer in self.layers:
            layer.compute_outputs(input)
            input = layer.out

    def _backward(self, X: np.ndarray, y: np.ndarray, lr: float):
        out_grads = self.loss_func.gradient(y, self.layers[-1].out)
        inputs = [X] + [layer.out for layer in self.layers[:-1]]
        for layer, input in zip(reversed(self.layers), reversed(inputs)):
            layer.compute_gradients(input, out_grads, batch=y.shape[0])
            layer.adjust_weights(lr)
            out_grads = out_grads @ layer.W

    def compute_loss(self, y: np.ndarray, y_pred: np.ndarray):
        return self.loss_func.function(y, y_pred)

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = X.copy()
        for layer in self.layers:
            out = layer.predict(out)
        return out

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epoch: int,
        batch: int,
        lr: float,
        X_test: np.ndarray,
        y_test: np.ndarray,
        result_path: str,
    ):
        if batch > X_train.shape[0]:
            raise Exception("batch size is bigger than training variables number.")
        for i in range(epoch):
            for j in range(0, X_train.shape[0], batch):
                s, e = j, min(j + batch, X_train.shape[0])
                self._forward(X_train[s:e])
                self._backward(X_train[s:e], y_train[s:e], lr)
            y_train_pred, y_test_pred = self.predict(X_train), self.predict(X_test)
            train_loss = self.compute_loss(y_train, y_train_pred)
            test_loss = self.compute_loss(y_test, y_test_pred)
            train_acc = accuracy_score(y_train, y_train_pred) * 100
            test_acc = accuracy_score(y_test, y_test_pred) * 100
            print(
                f"epoch : {i + 1}"
                + f"\n  Train loss: {train_loss:.2f} | Train acc.: {train_acc:.2f}"
                + f"\n  Test loss:  {test_loss:.2f} | Test acc.:  {test_acc:.2f}"
            )
            result = {
                "epoch": i + 1,
                "lr": lr,
                "batch": batch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
            }
            self.results.append(result)
            result_dir = Path(result_path).joinpath(f"micronn_{i + 1}")
            self.save_results(result_dir)

    def save_results(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        with open(path.joinpath("results.json"), "w") as f:
            json.dump(self.results, f, indent=2)
        with open(path.joinpath("layers.pkl"), "wb") as f:
            pickle.dump(self.layers, f)

    def load_layers(self, path: Path):
        if not path.exists:
            raise FileNotFoundError(f"File {path} does not exist.")
        with open(path, "rb") as f:
            return pickle.load(f)
