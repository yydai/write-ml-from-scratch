def loadDataSet(filename, algo='logistic'):
    X = []
    y = []
    with open(filename) as f:
        for line in f.readlines():
            lineArr = line.strip().split()
            if algo == 'logistic':
                x_d = [1.0, float(lineArr[0]), float(lineArr[1])]
            else:
                x_d = [float(lineArr[0]), float(lineArr[1])]
            X.append(x_d)
            y.append(int(lineArr[2]))
    f.close()
    return X, y
