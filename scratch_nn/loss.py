import numpy as np


def cross_entropy_loss(y: np.ndarray, y_predict: np.ndarray):
    return -np.sum(y * np.log(y_predict)) / y.shape[0]

def cross_entropy_gradient(y: np.ndarray, y_predict: np.ndarray):
    return y_predict - y

def l2_loss(y: np.ndarray, y_predict: np.ndarray):
    return np.sum(np.power(y_predict - y, 2)) / y.shape[0]

def l2_gradient(y: np.ndarray, y_predict: np.ndarray):
    return y_predict - y

def function(loss_type: str):
    match loss_type:
        case "l2_loss":
            return l2_loss
        case "cross_entropy":
            return cross_entropy_loss
        case _:
            raise Exception("This loss function is does not exists or is not supported.")

def gradient(loss_type: str):
    match loss_type:
        case "l2_loss":
            return l2_gradient
        case "cross_entropy":
            return cross_entropy_gradient
        case _:
            raise Exception("This loss function is does not exists or is not supported.")
