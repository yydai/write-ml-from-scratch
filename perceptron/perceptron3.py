import math
import numpy as np
import progressbar
from deeplearning.loss_functions import SquareLoss
from deeplearning.activation_functions import Sigmoid


class Perceptron():
    def __init__(self, n_iterations=20000,
                 activation_function=Sigmoid, loss=SquareLoss, learning_rate=0.01):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.loss = loss()
        self.activation_func = activation_function()

    def fit(self, X, y):
        n_samples, n_features = np.shape(X)
        _, n_outputs = np.shape(y)
        limit = 1 / math.sqrt(n_features)
        self.W = np.random.uniform(-limit, limit, (n_features, n_outputs))
        self.w0 = np.zeros((1, n_outputs))

        for i in self.progressbar(range(self.n_iterations)):
            linear_output = X.dot(self.W) + self.w0
