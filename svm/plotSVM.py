import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from tools import loadDataSet
from svmMliA import SmoSimple, calcWs, smoP


def loadPoint(train, labels):
    x_0 = []
    y_0 = []
    x_1 = []
    y_1 = []

    for i, v in enumerate(train):
        if labels[i] == 1:
            x_0.append(v[0])
            y_0.append(v[1])
        else:
            x_1.append(v[0])
            y_1.append(v[1])
    return x_0, y_0, x_1, y_1


def plotPoint(x_0, y_0, x_1, y_1, alphas, b, train, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # draw point
    ax.scatter(x_0, y_0, marker='s', s=90)
    ax.scatter(x_1, y_1, marker='o', s=50, c='red')
    plt.title('SVM Circled')

    # draw support vector circle
    for i, v in enumerate(alphas):
        if v:
            circle = Circle(tuple(train[i]), 0.3, facecolor='none', edgecolor=(
                0, 0.8, 0.8), linewidth=2, alpha=0.3)
            ax.add_patch(circle)
    w = calcWs(alphas, train, labels)
    # draw seperate line
    x = np.arange(-2.0, 12.0, 0.1)
    y = (-w[0] * x - b) / w[1]
    ax.plot(x, y)
    ax.axis([-2, 12, -8, 6])
    plt.show()


if __name__ == '__main__':
    train, labels = loadDataSet('testSet.txt')
    b, alphas = SmoSimple(train, labels, 0.2, 0.001, 40)
    x_0, y_0, x_1, y_1 = loadPoint(train, labels)
    plotPoint(x_0, y_0, x_1, y_1, alphas, b, train, labels)
