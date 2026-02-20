import numpy as np


def one_hot_encode(digit: int):
    if digit > 9:
        raise Exception("Give a digit between 0 and 9.")
    return [0] * digit + [1] + [0] * (9 - digit)


def accuracy_score(y: np.ndarray, y_pred: np.ndarray):
    return np.mean(np.argmax(y, axis=1) == np.argmax(y_pred, axis=1)) * 100


def manual_seed(seed):
    np.random.seed(seed)
