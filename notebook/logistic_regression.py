import sys
import numpy as np
import random
import matplotlib.pyplot as plt

sys.path.append("..")
from loss_function.loss_function import sigmoid
from utils.utils import loadDataSet


class LogisticRegression(object):
    def __init__(self, train_x, train_y, lr=0.001, maxIter=500):
        self.x = np.array(train_x)
        self.m = self.x.shape[0]
        self.n = self.x.shape[1]
        self.y = np.array(train_y).transpose().reshape((self.m, 1))
        self.w = np.ones((self.n, 1))
        self.lr = lr
        self.maxIter = maxIter

    def gradAscent(self):
        for k in range(self.maxIter):
            h = sigmoid(np.dot(self.x, self.w))
            error = (self.y - h)
            self.w = self.w + self.lr * np.dot(self.x.T, error)

    def stocGradAscent0(self):
        for i in range(self.m):
            h = sigmoid(np.dot(self.x[i], self.w))
            error = self.y[i] - h
            self.w += self.lr * self.x[i].reshape(3, 1) * error

    def stocGradAscent1(self):
        self.w = np.ones((self.n, 1))
        for i in range(self.maxIter):
            for j in range(self.m):
                self.lr = 4 / (1.0 + j + i) + 0.0001
                randIndex = int(random.uniform(0, len(self.x)))
                h = sigmoid(np.dot(self.x[randIndex], self.w))
                error = self.y[randIndex] - h
                self.w += self.lr * self.x[randIndex].reshape(3, 1) * error
                np.delete(self.x, randIndex)

    def classifyVector(self, tests):
        def filter_prob(x):
            if x > 0.5:
                return 1.0
            else:
                return 0.0
        tests = np.array(tests)
        m, n = tests.shape
        probs = sigmoid(np.dot(tests, self.w))
        print(probs)
        f = np.vectorize(filter_prob)
        return f(probs)

    def plot(self):
        xcord1 = []
        ycord1 = []
        xcord2 = []
        ycord2 = []
        for i in range(self.m):
            if int(self.y[i]) == 1:
                xcord1.append(self.x[i, 1])
                ycord1.append(self.x[i, 2])
            else:
                xcord2.append(self.x[i, 1])
                ycord2.append(self.x[i, 2])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
        ax.scatter(xcord2, ycord2, s=30, c='green')
        x = np.arange(-3.0, 3.0, 0.1)
        y = (-self.w[0] - self.w[1] * x) / self.w[2]
        ax.plot(x, y)
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.show()


if __name__ == '__main__':
    x, y = loadDataSet('logistic_regression_data.txt')
    lr = LogisticRegression(x, y)

    lr.stocGradAscent1()
    # lr.plot()

    print(lr.classifyVector(
        [[1, 0.569411, 9.548755], ]))
