import numpy as np


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


def hinge(z):
    return max(0, 1 - z)
