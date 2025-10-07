import numpy as np

from scratch_nn import activations


class Layer:
    def __init__(self, neuron_number: int, activation_function: str):
        self.neuron_number = neuron_number
        self.activation_function = activation_function

    def initialize_weights(self, input_len: int) -> np.ndarray:
        # The input length (input_len) should already include the bias
        self.weights = np.random.uniform(-1.0, 1.0, (self.neuron_number, input_len))
        return self.weights

    def compute_outputs(self, X: np.ndarray) -> np.ndarray:
        activate = activations.function(self.activation_function)
        out = activate(self.weights @ X.T)
        self.outputs = out.T # Transpose to make like y
        return self.outputs

    def compute_errors(self, out_gradients: np.ndarray) -> np.ndarray:
        compute = activations.error(self.activation_function)
        err = compute(self.outputs, out_gradients)
        return err

    def adjust_weights(self, errors: np.ndarray, input: np.ndarray, learning_rate: float) -> np.ndarray:
        if self.outputs.size == 0:
            raise Exception("You must calculate the outputs before calculating the derivatives.")

        self.gradients = input @ errors
        self.weights -= learning_rate * (self.gradients.T / len(input[0]))
        return self.weights
