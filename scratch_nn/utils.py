import numpy as np


def one_hot_encode(digit: np.ndarray):
    vec = np.zeros(10, dtype=int)
    vec[digit] = 1
    return vec


def accuracy_score(y: np.ndarray, y_pred: np.ndarray):
    y_true_classes = np.argmax(y, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    correct = (y_true_classes == y_pred_classes)
    accuracy = np.mean(correct)
    return accuracy
