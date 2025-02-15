# MNIST Neural Network from Scratch

This repository contains a Jupyter notebook that demonstrates how to implement a simple two-layer neural network from scratch and train it on the MNIST digit recognizer dataset. The goal of this workshop is to provide an instructional example through which you can understand the basic concepts of neural networks, including forward and backward propagation, parameter initialization, and gradient descent.
Godspeed Randomize();

## Repository Contents

- **mnist-scratch.ipynb**: The primary notebook containing the implementation of the neural network.
- **train.csv**: The dataset used for training the neural network, which consists of images of handwritten digits.

## Workshop Overview

### 1. Introduction

We start by importing the necessary libraries and loading the MNIST dataset.

```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('train.csv')
```

### 2. Data Preparation

The dataset is shuffled and split into training and development sets. The pixel values are normalized to be between 0 and 1.

```python
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

from sklearn.model_selection import train_test_split

X = data[:, 1:]  # Features
Y = data[:, 0]   # Labels

X_train, X_dev, Y_train, Y_dev = train_test_split(X, Y, test_size=1000, random_state=42)

X_train = X_train.T / 255.
X_dev = X_dev.T / 255.
```

### 3. Neural Network Implementation

We define functions for initializing parameters, forward propagation, activation functions (ReLU and Softmax), backward propagation, and parameter updates.

```python
def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    return A

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2
```

### 4. Training the Neural Network

We train the neural network using gradient descent and print the accuracy at regular intervals.

```python
def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.12, 500)
```

### 5. Testing the Neural Network

We test the neural network on the development set and print the accuracy.

```python
def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
print("Accuracy on dev set:", get_accuracy(dev_predictions, Y_dev))
```

## Results

The neural network achieves approximately 85% accuracy on the training set and 85.2% accuracy on the development set.

## Dependencies

- numpy
- pandas
- matplotlib
- scikit-learn

To install the required dependencies, run the following command:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Usage

To run the notebook, open it in Jupyter and execute the cells in order. The notebook will guide you through the process of training and testing the neural network on the MNIST dataset.

## Conclusion

This workshop demonstrated how to build a simple neural network from scratch and train it on the MNIST dataset. By understanding the implementation details, you can gain a deeper understanding of how neural networks work and apply these concepts to more complex problems.

Feel free to experiment with different network architectures, hyperparameters, and datasets to further enhance your understanding of neural networks.

Happy learning!
