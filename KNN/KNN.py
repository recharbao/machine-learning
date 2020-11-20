#-*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import numpy as np
from collections import Counter

def KNN(inputData, data, dataClass, k):
    data = np.array(data)
    labels = np.unique(dataClass)
    labelsIndex = np.zeros((labels.shape))
    
    diff = data - inputData
    dis = np.sum(diff ** 2, axis=1)
    result = []
    for i in range(k):
        index = np.argmin(dis)
        result.append(index)
        dis[index] = sys.maxsize

    curLabels = [dataClass[i] for i in result]
    result = Counter(curLabels)
    return dict(result.most_common(1)).keys()



data = [[2, 7], [1, 3], [2, 2], [4, 2], [7, 2], [0, 0], [5, 5], [3, 3], [4, 6]]
dataClass = ['red','red', 'red', 'red', 'blue', 'blue', 'blue', 'blue', 'blue']


nums = np.max(data) + 5

for i in range(nums):
    for j in range(nums):
        if [i, j] in data:
            if dataClass[data.index([i, j])] == 'red':
                print '+',
            else:
                print '*',
        else:
            print '.',
    print('')


print('\n\n\n')

inputData = [2, 5]

for i in range(nums):
    for j in range(nums):
        if [i, j] in data:
            if dataClass[data.index([i, j])] == 'red':
                print '+',
            else:
                print '*',
        elif [i, j] == inputData:
            print '$',
        else:
            print '.',
    print('')



result = KNN(inputData, data, dataClass, 3)

print('class:', result)

