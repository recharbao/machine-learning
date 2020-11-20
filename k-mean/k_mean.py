#-*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import numpy as np
import random

#初始化簇中心
def InitClusterCenter(data, k):
    centerIndex = []
    centerPoint = []
    num = len(data) - 1

    while len(centerIndex) < k:
        index = random.randint(0, num)
        if index not in centerIndex:
            centerIndex.append(index)

    centerPoint = [data[i] for i in centerIndex]

    return centerPoint


#求距离
#欧式距离
def Distance(dataOneLine, center):
    return np.sqrt(np.sum((dataOneLine - center) ** 2))


#迭代
# N是最大迭代次数
def k_mean(data, k, N):

    data = np.array(data)
    #初始化
    centerPoint = InitClusterCenter(data, k)

    nums = data.shape[0]
    
    dim = data.shape[1]

    dataCLass = np.zeros((nums))

    #开始迭代
    for i in range(N):
        for j in range(nums):
            minDis = sys.maxsize
            minIndex = -1
            for l in range(k):
                if Distance(data[j], centerPoint[l]) < minDis:
                    minDis = Distance(data[j], centerPoint[l])
                    minIndex = l
            dataCLass[j] = minIndex

        #更新聚簇
        for l in range(k):
            datal = np.array([i for i in range(nums) if dataCLass[i] == l])
            datall = np.array([data[i] for i in datal])
            centerPoint[l] = np.mean(datall, axis=0)

    return centerPoint, dataCLass


data = [[1,1],[2,1],[1,2],[2,2],[4,3],[5,3],[4,4],[5,4]]

centerPoint = InitClusterCenter(data, 3)

centerPoint, dataClass = k_mean(data, 3, 50)

print(centerPoint)


