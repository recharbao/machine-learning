#-*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import numpy as np
import pickle
import matplotlib.pyplot as plt


class SVM(object):
    def __init__(self):
        self.W = None

    #计算梯度
    def lossGradient(X, y, reg):

        #计算loss
        nums_X = X.shape[0]
        scores = np.dot(self.W, X.T)
        correct_scores = scores[list(y), range(nums_X)]
        diff = np.max(scores - correct_scores + 1) #delta = 1
        diff[list(y), range(nums_X)] = 0
        loss = 1/nums_X * np.sum(diff) + 0.5 * reg * np.sum(diff * diff)

        #计算dw
        dw = np.zeros((self.W.shape))
        nums_class = self.W.shape[0]
        mark = np.zeros((nums_class, nums_X))
        mark[diff > 0] = 1
        mark[list(y), range(nums_X)] = 0
        mark[list(y), range(nums_X)] -= np.sum(mark, axis=1)nb
        dW = np.dot(mark, X.T)
        dW = dW / num_train + reg * W



    #训练
    def train(X_tr, y_tr, batch_size = 200, iter_nums, reg):

        num_train = X_tr.shape[0]
        dataDim = X_tr.shape[1] + 1 #因为添加了偏置
        classNum = np.max(y) + 1  #该训练集为0～9号labels

        if self.W == None:
            #正态分布随机
            self.W = 0.001 * np.random.randn(classNum, dataDim)

        #进行n次迭代
        for i in range(iter_nums):
            index_batch = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X_tr[index_batch]
            y_batch = y_tr[index_batch]
            loss, dw = self.lossGradient(X_batch, y_batch, reg)
            






#根据官网指示获取数据
def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict


def loadCIFAR():
    X_train = []
    y_train = []
    for i in range(1, 6):
        pathData = '/media/rechar/新加卷/资料/cifar-10-batches-py/data_batch_%d'%(i)
        data_batch_dict = unpickle(pathData)
        X_train.append(data_batch_dict['data'].astype('float'))
        y_train.append(data_batch_dict['labels'])

    

    pathTest = '/media/rechar/新加卷/资料/cifar-10-batches-py/test_batch'
    test_batch_dict = unpickle(pathTest)
    X_test = test_batch_dict['data']
    y_test = test_batch_dict['labels']


    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)


X_tr, y_tr, X_te, y_te = loadCIFAR()

'''
#修改shape来输出图像
X_train = X_tr.reshape(50000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
y_train = y_tr.reshape(50000)
X_test = X_te.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
y_test = y_te.reshape(10000)


dataClass = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#在每个类别中分别选出10幅图像
for i, j in enumerate(dataClass):
    picIndex = np.flatnonzero(y_train == i)
    pic = np.random.choice(picIndex, 6, replace=False) #不重复
    for x, y in enumerate(pic):
        plt_id = x * 10 + (i + 1)
        plt.subplot(6, 10, plt_id)
        plt.imshow(X_train[y])
        #去掉下标
        plt.axis('off')
        if x == 0:
            plt.title(j)
plt.show()
'''

#增加偏置
onesTr = np.ones((X_tr.shape[0]))
onesTe = np.ones((X_te.shape[0]))

X_tr = np.insert(X_tr, X_tr.shape[1], values=onesTr, axis=1)
X_te = np.insert(X_te, X_te.shape[1], values=onesTe, axis=1)











'''

learning_rates = [1.4e-7, 1.5e-7, 1.6e-7]
regularization_strengths = [8000.0, 9000.0, 10000.0, 11000.0, 18000.0, 19000.0, 20000.0, 21000.0]


for lr in learning_rates:
    for reg in regularization_strengths:


'''


