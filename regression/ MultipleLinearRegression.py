#-*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import numpy as np

class MLR(object):
    def __init__(self):
        self.B = None

    def MultipleLinearRegression(self, X, y):
        self.B = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        return self.B
    
    def predict(self, x):
        return self.B.T.dot(x.T)


if __name__ == "__main__":

    X = np.array([[1, 20, 400],
                [1, 25, 625],
                [1, 30, 900],
                [1, 35, 1225],
                [1, 40, 1600],
                [1, 50, 2500],
                [1, 60, 3600],
                [1, 65, 4225],
                [1, 70, 4900],
                [1, 75, 5625],
                [1, 80, 6400],
                [1, 90, 8100]])

    y = [1.81, 1.70, 1.65, 1.55, 1.48, 1.40, 1.30, 1.26, 1.24, 1.21, 1.20, 1.18]

    mlr = MLR()
    B = mlr.MultipleLinearRegression(X, y)
    print(B)

    x = np.array([1, 100, 10000])
    pred = mlr.predict(x)
    print(pred)


