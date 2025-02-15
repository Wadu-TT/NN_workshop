# MNIST Neural Network from Scratch

This repository contains a Jupyter notebook that demonstrates how to implement a simple two-layer neural network from scratch and train it on the MNIST digit recognizer dataset. The goal of this workshop is to provide an instructional example through which you can understand the basic concepts of neural networks, including forward and backward propagation, parameter initialization, and gradient descent.

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
 ▋
