import numpy as np

from tools import loadDataSet, selectJrand, clipAlpha


def SmoSimple(train, classLabels, C, toler, maxIter):
    train = np.array(train)
    m, n = train.shape
    labels = np.array(classLabels).transpose().reshape(m, 1)
    b = 0
    alphas = np.zeros((m, 1))
    iter = 0
    while (iter < maxIter):
        alphaChanged = 0
        for i in range(m):
            # f(x) = wx + b = alpha_j * lable_j * x_j * x_i
            fx_i = float(np.dot((alphas * labels).T,
                                (np.dot(train, train[i, :])).T)) + b
            E_i = fx_i - float(labels[i])
            r2 = E_i * float(labels[i])
            if ((r2 < -toler) and (alphas[i] < C) or ((r2 > toler) and (alphas[i] > 0))):
                j = selectJrand(i, m)
                fx_j = float(np.dot((alphas * labels).T,
                                    (np.dot(train, train[j, :])).T)) + b
                E_j = fx_j - float(labels[j])
                alpha_i_old = alphas[i].copy()
                alpha_j_old = alphas[j].copy()
                # get the lower bound and upper bound
                if (labels[i] != labels[j]):
                    # L H
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])

                if L == H:
                    continue
                Kii = np.dot(train[i, :], train[i, :].T)
                Kjj = np.dot(train[j, :], train[j, :].T)
                Kij = np.dot(train[i, :], train[j, :].T)
                eta = Kii + Kjj - 2.0 * Kij
                if eta <= 0:
                    continue
                alphas[j] += labels[j] * (E_i - E_j) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)

                if abs(alphas[j] - alpha_j_old) < 0.00001:
                    continue

                # alphas old plus == alphas new plus
                alphas[i] = alphas[i] + labels[i] * \
                    labels[j] * (alpha_j_old - alphas[j])
                b1 = b - E_i - labels[i] * (alphas[i] - alpha_i_old) * \
                    Kii - labels[j] * Kij * (alphas[j] - alpha_j_old)
                b2 = b - E_j - labels[i] * (alphas[i] - alpha_i_old) * \
                    Kij - labels[j] * Kjj * (alphas[j] - alpha_j_old)

                if ((alphas[i] > 0) and (alphas[i] < C)):
                    b = b1
                elif ((alphas[j] > 0) and (alphas[j] < C)):
                    b = b2
                else:
                    b = (b1 + b2) / 2
                alphaChanged += 1
        if (alphaChanged == 0):
            iter += 1
        else:
            iter = 0
        print("interation number:", iter)
        print(b)
    return b, alphas


def calcWs(alphas, dataArr, classLabels):
    train = np.array(dataArr)
    m, n = train.shape
    labels = np.array(classLabels).transpose().reshape(m, 1)
    w = np.dot((alphas * labels).T, train)
    return w.reshape((-1, )).tolist()


class optStruct(object):
    '''
    Another version of svm, but more faster than the simple one
    '''

    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn
        self.C = C
        self.tol = toler
        self.m = dataMatIn.shape[0]
        self.labels = classLabels.reshape(self.m, 1)
        self.alphas = np.array(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.array(np.zeros((self.m, 2)))

    def calcEk(self, k):
        fx_k = np.dot((self.alphas * self.labels).T,
                      (np.dot(self.X, self.X[k, :])).T) + self.b
        E_k = fx_k - float(self.labels[k])
        return E_k

    def selectJ(self, i, E_i):
        maxK = -1
        maxDeltaE = 0
        E_j = 0
        self.eCache[i] = [1, E_i]
        valideCacheList = np.nonzero(self.eCache[:, 0])[0]
        if (len(valideCacheList) > 1):
            for k in valideCacheList:
                if k == i:
                    continue
                E_k = self.calcEk(k)
                deltaE = abs(E_i - E_k)
                if (deltaE > maxDeltaE):
                    maxK = k
                    maxDeltaE = deltaE
                    E_j = E_k
            return maxK, E_j
        else:
            j = selectJrand(i, self.m)
            E_j = self.calcEk(j)
        return j, E_j

    def updateEk(self, k):
        E_k = self.calcEk(k)
        self.eCache[k] = [1, E_k]

    def innerL(self, i):
        E_i = self.calcEk(i)
        if ((self.labels[i] * E_i < -self.tol) and (self.alphas[i] < self.C)) or \
                ((self.labels[i] > self.tol) and (self.alphas[i] > 0)):
            j, E_j = self.selectJ(i, E_i)
            alpha_i_old = self.alphas[i].copy()
            alpha_j_old = self.alphas[j].copy()
            if (self.labels[i] != self.labels[j]):
                L = max(0, self.alphas[j] - self.alphas[i])
                H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
            else:
                L = max(0, self.alphas[j] + self.alphas[i] - self.C)
                H = min(self.C, self.alphas[j] + self.alphas[i])
            if L == H:
                return 0
            Kii = np.dot(self.X[i, :], self.X[i, :].T)
            Kjj = np.dot(self.X[j, :], self.X[j, :].T)
            Kij = np.dot(self.X[i, :], self.X[j, :].T)
            eta = Kii + Kjj - 2.0 * Kij
            if eta <= 0:
                return 0
            self.alphas[j] += self.labels[j] * (E_i - E_j) / eta
            self.alphas[j] = clipAlpha(self.alphas[j], H, L)
            self.updateEk(j)
            if (abs(self.alphas[j] - alpha_j_old) < 0.00001):
                return 0
            self.alphas[i] = self.alphas[i] + self.labels[i] * \
                self.labels[j] * (alpha_j_old - self.alphas[j])
            b1 = self.b - E_i - self.labels[i] * (self.alphas[i] - alpha_i_old) * \
                Kii - self.labels[j] * Kij * (self.alphas[j] - alpha_j_old)
            b2 = self.b - E_j - self.labels[i] * (self.alphas[i] - alpha_i_old) * \
                Kij - self.labels[j] * Kjj * (self.alphas[j] - alpha_j_old)
            if (self.alphas[i] > 0) and (self.alphas[i] < self.C):
                self.b = b1
            elif (self.alphas[j] > 0) and (self.alphas[j] < self.C):
                self.b = b2
            else:
                self.b = (b1 + b2) / 2
            return 1
        else:
            return 0


def smoP(dataMatIn, classLabels, C, toler, maxIter):
    opt = optStruct(np.array(dataMatIn), np.array(
        classLabels).transpose(), C, toler)
    iter = 0
    entireSet = True
    alphaChanged = 0
    while (iter < maxIter) and ((alphaChanged > 0) or (entireSet)):
        alphaChanged = 0
        if entireSet:
            for i in range(opt.m):
                alphaChanged += opt.innerL(i)
            iter += 1
        else:
            nonBoundIs = np.nonzero((opt.alphas > 0) * (opt.alphas < opt.C))[0]
            for i in nonBoundIs:
                alphaChanged += opt.innerL(i)
            iter += 1
        if entireSet:
            entireSet = False
        elif (alphaChanged == 0):
            entireSet = True

    return opt.b, opt.alphas


if __name__ == '__main__':
    train, labels = loadDataSet('testSet.txt')
    b, alphas = smoP(train, labels, 0.6, 0.01, 40)
    print(alphas[alphas > 0])
    # print (calcWs(alphas, train, labels))
