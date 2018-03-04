import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [3, 3],
    [4, 3],
    [1, 1]
])

y = np.array([1, 1, -1])

for i in range(len(y)):
    if y[i] == 1:
        plt.plot(X[i][0], X[i][1], 'bo')
    else:
        plt.plot(X[i][0], X[i][1], 'rx')


def perceptron_sgd(X, Y):
    w = np.zeros(len(X[0]))
    b = 0
    learning_rate = 1
    epochs = 9

    for i in range(epochs):
        for j, x in enumerate(X):
            if ((np.dot(w, X[j]) + b) * Y[j]) <= 0:
                w = w + learning_rate * X[j] * Y[j]
                b = b + learning_rate * Y[j]
                break
        print w, b
    return w, b


def perceptron_sgd_2(X, Y):

    # 1. get Gram matrix
    # 2. alpha <- 0 and b <- 0
    # 3. (x1, y1)
    pass


perceptron_sgd(X, y)
