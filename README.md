# mnist-perceptron

A simple perceptron training and prediction tool for the MNIST dataset, coded from scratch using only NumPy and the Python standard library.

## ⚠️ Warning

The module scratch_nn used here is not intended for any use other than learning. I only programmed this module for myself to learn about neural networks and backpropagation.

## 📊 Results

Using scratch_nn, I got decent results, contained in out/final_results.txt:

* 0.29 validation loss

* 86.18% validation accuracy

The file out/final_results.pkl contains the data obtained from training. It's a dictionary that stores the epoch, learning rate, batch size, loss function, and the layers containing the weights of the trained neural network.

## ✅ Testing

The trained model can be tested by copying out/final_results.pkl to out/results.pkl and using similar code to what is in model_training.py.

The .predict() method can be used to compute outputs, and accuracy_score() from utils.py can be used to compute the accuracy score.
