import numpy as np
from keras.datasets import mnist

from scratch_nn.layer import Layer
from scratch_nn.nn import NN
from scratch_nn.utils import one_hot_encode

print("Loading dataset...")
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("Data loaded successfully.")
print(f"\nX_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}\n")


print("Transforming data...")
X_train_flat = X_train.reshape(X_train.shape[0], -1)
y_train_enc = np.array([np.array(one_hot_encode(y)) for y in y_train])
X_test_flat = X_test.reshape(X_test.shape[0], -1)
y_test_enc = np.array([np.array(one_hot_encode(y)) for y in y_test])

print("Data transformation completed successfully.")
print(f"\nFlattened X_train shape: {X_train_flat.shape}")
print(f"Encoded y_train shape: {y_train_enc.shape}")
print(f"\nFlattened X_test shape: {X_test_flat.shape}")
print(f"Encoded y_test shape: {y_test_enc.shape}")

print("Setting model...")
nn = NN(784, [Layer(10, "sigmoid")], loss='l2_loss')
print("Model setted.")

print("Training model...")
nn.train(
    epoch=10,
    learning_rate=0.01,
    batch_size=128,
    X_train=X_train_flat,
    y_train=y_train_enc,
    X_test=X_test_flat,
    y_test=y_test_enc
)
print("Training finished.")
