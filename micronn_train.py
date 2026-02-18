import numpy as np

from micronn.activations import Softmax
from micronn.layer import Layer
from micronn.loss import CrossEntropy
from micronn.nn import NN
from micronn.utils import one_hot_encode

print("Loading dataset...")
data = np.load("data/mnist.npz")
X_train, y_train = data["x_train"], data["y_train"]
X_test, y_test = data["x_test"], data["y_test"]

print("Preprocess data...")
X_train_flat = X_train.reshape(X_train.shape[0], -1)
y_train_enc = np.array([np.array(one_hot_encode(y)) for y in y_train])
X_test_flat = X_test.reshape(X_test.shape[0], -1)
y_test_enc = np.array([np.array(one_hot_encode(y)) for y in y_test])

print("Setting model...")
nn = NN([Layer(10, Softmax(), input_size=784)], CrossEntropy())

print("Training model...")
nn.train(
    X_train_flat,
    y_train_enc,
    epoch=5,
    batch=128,
    lr=1e-2,
    X_test=X_test_flat,
    y_test=y_test_enc,
    result_path="models/",
)

print("Training finished successfully!")
