import numpy as np

import micronn.layer as l
from micronn.loss import CrossEntropyLoss
from micronn.nn import NN
from micronn.utils import manual_seed, one_hot_encode


def process_data(path: str):
    data = np.load(path)
    X_train, y_train = data["x_train"], data["y_train"]
    X_test, y_test = data["x_test"], data["y_test"]

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    y_train_enc = np.array([np.array(one_hot_encode(y)) for y in y_train])
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    y_test_enc = np.array([np.array(one_hot_encode(y)) for y in y_test])
    return {
        "x_train": X_train_flat,
        "y_train": y_train_enc,
        "x_test": X_test_flat,
        "y_test": y_test_enc,
    }


def main():
    print("Preprocess data...")
    data = process_data("data/mnist.npz")
    X_train, y_train = data["x_train"], data["y_train"]
    X_test, y_test = data["x_test"], data["y_test"]
    print("Setting model...")
    manual_seed(42)
    nn = NN(
        [l.LinearLayer(size=10, input_size=784), l.SoftmaxLayer()],
        CrossEntropyLoss(),
    )
    print("Training model...")
    nn.train(
        X_train,
        y_train,
        epochs=6,
        batch=128,
        lr=1e-2,
        X_test=X_test,
        y_test=y_test,
        result_path="models",
    )
    print("Training finished successfully!")


if __name__ == "__main__":
    main()
