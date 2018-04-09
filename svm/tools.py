import random
import numpy as np


def loadDataSet(filename):
    dataMat = []
    labelMat = []
    f = open(filename)
    for line in f.readlines():
        data = line.strip().split('\t')
        data = list(map(float, data))
        dataMat.append([data[0], data[1]])
        labelMat.append(data[2])
    return dataMat, labelMat


def selectJrand(i, m):
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j

# H high value L low value, aj in [H, L]


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj


if __name__ == '__main__':
    a, b = loadDataSet('testSet.txt')
    print(np.mat(b).transpose())
