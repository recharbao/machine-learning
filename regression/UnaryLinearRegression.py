#-*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import numpy as np



class LR(object):
    def __init__(self):
        self.b0 = None
        self.b1 = None
        
    #采用最小二乘方法
    def LinearRegression(self, X, y):
        n = float(len(X))
        Sx = np.sum(X)
        Sy = np.sum(y)
        Sx_2 = np.sum(X ** 2)
        Sy_2 = np.sum(y ** 2)
        Sxy = np.sum(X * y)

        self.b1 = (Sxy - (1 / n) * (Sx * Sy)) / (Sx_2 - (1 / n) * (Sx ** 2))
        self.b0 = (1 / n) * (Sy) - (1 / n) * (Sx) * self.b1

        return self.b1, self.b0


    def predict(self, x):
        return self.b0 + x * b1



if __name__ == "__main__":
    
    X = np.linspace(100, 190, 10)
    y = np.array([45, 51, 54, 61, 66, 70, 74, 78, 85, 89])

    lr = LR()
    b1, b0 = lr.LinearRegression(X, y)
    print(b1)
    print(b0)
    x = input("输入一个数值:")
    pred = lr.predict(x)
    print(pred)


