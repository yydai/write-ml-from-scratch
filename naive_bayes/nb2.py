import numpy as np
import random
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def load_train_data(filename):
    train_x = []
    train_y = []
    with open(filename) as f:
        for line in f.readlines():
            lineArr = line.strip().split()
            train_x.append(lineArr[:2])
            train_y.append(lineArr[-1])
        return train_x, train_y


def train(train_set, train_labels, class_num):
    train = np.array(train_set)
    m, n = train.shape
    prob = np.zeros((class_num, 2))
    cp = np.zeros((class_num, n, 2))
    labels = {}
    for k, v in enumerate(train_labels):
        labels.setdefault(v, 0)
        labels[v] += 1
    for k, v in enumerate(labels):
        prob[k] = v, labels[v] / float(m)

    # caculate conditional_probability
    for v in range(prob.shape[0]):
        cp[]


if __name__ == '__main__':
    class_number = 2  # 1 -1
    filename = 'train_data.txt'  # from Li Hang book
    train_x, train_y = load_train_data(filename)
    train(train_x, train_y, class_number)
