import os
import pickle

import numpy as np

from scratch_nn import loss
from scratch_nn.layer import Layer
from scratch_nn.utils import accuracy_score


class NN:
    "A class for a neural network model"

    def __init__(self, X_len: int, layers: list[Layer] = [], loss: str = "l2_loss"):
        self.X_len = X_len
        self.layers = layers
        self.loss = loss
        self.epoch = 0
        self.result_path = "out/results.pkl"
        self._initialize_layers()

    def _initialize_layers(self):
        """Initialize the layers of the neural network"""
        print("Initializing layers...")
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if i == 0:
                layer.initialize_weights(self.X_len + 1)
            else:
                inputs = self.layers[i - 1].neuron_number + 1
                layer.initialize_weights(inputs)

        if os.path.exists(self.result_path):
            try:
                print("Reading layers from cache...")
                self.load_layers(self.result_path)
            except Exception as e:
                print("Error while loading layers: ", e)

    def _add_bias(self, X: np.ndarray):
        return np.concatenate((np.ones((X.shape[0], 1), dtype=float), X), axis=1)

    def _forward(self, X: np.ndarray):
        """Return the result of forward propagation of the neural network."""
        for i in range(len(self.layers)):
            if i == 0:
                prev_layer = X
            else:
                prev_layer = self.layers[i - 1].outputs

            prev_layer_prime = self._add_bias(prev_layer)
            self.layers[i].compute_outputs(prev_layer_prime)

        return self.layers[-1].outputs

    def compute_loss(self, y: np.ndarray, y_pred: np.ndarray):
        """Compute the loss according to the loss function."""
        loss_function = loss.function(self.loss)
        return loss_function(y, y_pred)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._forward(X)

    def _backward(self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray, learning_rate: float):
        """Adjust the neurons weights according to backpropagation algorithm
        for regression."""

        loss_gradient = loss.gradient(self.loss)
        errors = loss_gradient(y, y_pred)

        for i in range(len(self.layers) - 1, -1, -1):
            self.layers[i].compute_errors(errors)

            if i == 0:
                input = X
            else:
                input = self.layers[i - 1].outputs

            input_prime = self._add_bias(input).T
            self.layers[i].adjust_weights(errors, input_prime, learning_rate)
            errors = errors @ np.delete(self.layers[i].weights, 1, axis=1)

    def train(self, epoch: int, learning_rate: float, batch_size: int, X_train: np.ndarray,
              y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):

        self.learning_rate = learning_rate
        self.batch_size = batch_size

        for i in range(epoch):
            for j in range(0, X_train.shape[0], batch_size):
                b, e = j, min(j + batch_size, X_train.shape[0])
                y_pred = self._forward(X_train[b:e])
                self._backward(X_train[b:e], y_train[b:e], y_pred, learning_rate)

            self.epoch += 1
            y_train_pred = self.predict(X_train)
            train_loss = self.compute_loss(y_train, y_train_pred)
            y_test_pred = self.predict(X_test)
            test_loss = self.compute_loss(y_test, y_test_pred)
            print(f"""epoch : {self.epoch}
Train loss: {train_loss} | Train acc: {accuracy_score(y_train, y_train_pred)}
Test loss: {test_loss} | Test acc: {accuracy_score(y_test, y_test_pred)}""")

            if self.layers:
                self.save_layers(self.result_path)
            else:
                print("Corrupted layers not cached.")

    def save_layers(self, path: str):
        with open(path, 'wb') as f:
            result = {
                "epoch": self.epoch,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "loss": self.loss,
                "layers": self.layers,
            }
            pickle.dump(result, f)

    def load_layers(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError()

        with open(path, 'rb') as f:
            result = pickle.load(f)
            self.epoch = result["epoch"]
            self.learning_rate = result["learning_rate"]
            self.batch_size = result["batch_size"]
            self.loss = result["loss"]
            self.layers = result["layers"]
