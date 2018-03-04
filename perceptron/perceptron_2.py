import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, x, y=1):
        self.x = x
        self.y = y
        self.alpha = np.zeros((self.x.shape[0], 1))
        self.b = 0.0
        self.learning_rate = 1

        self.gram = np.zeros((self.x.shape[0], self.x.shape[0]))
        # contruct gram
        for i, k1 in enumerate(self.x):
            for j, k2 in enumerate(self.x):
                self.gram[i][j] = np.dot(k1, k2)

    def train(self):
        length = self.x.shape[0]
        while True:
            count = 0
            for i in range(length):
                print self.alpha * self.y
                print self.gram[i]
                y = np.dot(self.gram[i],
                           self.alpha * self.y) + self.b

                if y * self.y[i] <= 0:
                    self.alpha[i] = self.alpha[i] + self.learning_rate
                    self.b = self.b + self.y[i] * self.learning_rate
                    count += 1
            if count == 0:
                return np.sum(self.x * self.alpha * self.y, axis=0), self.b


class ShowPicture:
    def __init__(self, x, y, w, b):
        self.b = b
        self.w = w
        plt.figure(1)
        plt.title('test', size=14)
        plt.xlabel('x-axis', size=14)
        plt.ylabel('y-axis', size=14)

        xData = np.linspace(0, 5, 100)
        yData = self.expression(xData)
        plt.plot(xData, yData, color='r', label='y1 data')

        for i in range(x.shape[0]):
            if y[i] < 0:
                plt.scatter(x[i][0], x[i][1], marker='x', s=50)
            else:
                plt.scatter(x[i][0], x[i][1], s=50)

    def expression(self, x):
        y = (-self.b - self.w[0] * x) / self.w[1]
        return y

    def show(self):
        plt.show()


xArray = np.array([
    [3, 3],
    [3, 4],
    [1, 1]
])

yArray = np.array([
    [1], [1], [-1]
])

p = Perceptron(xArray, yArray)
w, b = p.train()
print w, b
s = ShowPicture(xArray, yArray, w, b)
s.show()
