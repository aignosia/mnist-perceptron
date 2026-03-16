# MicroNN

MicroNN (abbreviation of Micro Neural Network) is a simple perceptron
training and inference script for simple neural networks. It leverages only
NumPy's fast computation for matrix operations, but each matrix operation
can be coded manually if wanted.

The `micronn.py` file is a script that can be used to train a one
layer perceptron on the MNIST dataset using softmax as activation layer and
cross-entropy as loss function. There is also the `pytorch_train.py` file
which is a script for training the same model used for benchmark.

Warning: I created this repository to learn about neural network and
backpropagation and do not intend to use it for any purpose other than
learning.

## Results

The results of training are available inside the `models` directory. Inside
`models`, there are two directories: `micronn` contains the model trained
using MicroNN and `pytorch` for the benchmark model trained using PyTorch.

Each directory contains a JSON file which contains the hyperparameters and
metrics obtained from training and a `.pkl` file containing the models
parameters.

The two models are trained using the same hyperparameters. However, the
PyTorch training achieves better results (92.03% test accuracy) with fewer
epochs compared to the MicroNN training (87.61% test accuracy). This is
maybe due to better initialization. But the results show that MicroNN
can achieve good results.

## Testing

### Prerequisites

- Python 3.11 (preferred) or above

### Installation

- Cloning the repo

```bash
git clone https://github.com/aignosia/mnist-perceptron.git
```

- Creating and activating virtual environment

```bash
# uv
uv venv
# Other Python installation
python -m venv .venv

source .venv/bin/activate
```

- Installing dependencies

```bash
# uv
uv sync --locked
# Other Python installation
pip install -r requirements.txt
```

If you do not have GPU available or don't want to use GPU, install NumPy
and PyTorch manually.

### Inference

The trained model can be tested by instantiating a `NN()` object, loading
the `.pkl` file with pickle then using `forward()` to
make inference on your data.

Example of testing script :

```python
# inference.py
import numpy as np
from micronn import NN, forward

# Load your data here

model = pickle.load("models/micronn/model.pkl")

y_logits = forward(X).layers[-1].out # X is an array of shape (784, 1)
y_pred = softmax(y_logits).argmax(axis=1)
print(y_pred)
```

### Replicating

You can replicate the results obtained by downloading a NPZ version of the
MNIST dataset, copying it to `data/mnist.npz` and by running the
`micronn.py` script.
