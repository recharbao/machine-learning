
from numpy.lib.scimath import power
from math import sqrt
import random
from numpy.core import zeros
from numpy import mean, nonzero
import numpy as np

#计算距离
def getdistance(v1,v2):
    return sqrt(sum(power(v1 - v2,2)))


#初始化随机k个簇中心
def initClusterCenter(data,k):
    #数据的数量，方便下面随机选择数据作为簇中心
    num = data.shape[0]
    data_dim = data.shape[1]
    centerpoint = zeros((k,data_dim))
    #查看是否重复选择
    check = []
    for i in range(k):
        flag = False;
        while not(flag):
            index = int(random.randint(0, num-1))
            if index not in check:
                check.append(index)
                flag = True
        print("index=")
        print(index)
        centerpoint[i,:] = data[index,:]
    return centerpoint

def kmeans(data, k):
    
    num = data.shape[0]
    dataClass = zeros((num,2))
    dataClassChange = True


    centerpoint = initClusterCenter(data,k)
    print("初始簇中心：")
    print(centerpoint)

    while dataClassChange:
        dataClassChange = False;
        for i in range(num):
            #找出每一个sample距离最小的簇中心
            minDistance = 999999999;
            minIndex = 0

            for j in range(k):
                #找最近
                distance = getdistance(centerpoint[j,:],data[i,:])
                if distance < minDistance:
                    minDistance = distance
                    minIndex = j

            #更换该带点所属的族类别
            if dataClass[i,0] != minIndex:
                dataClassChange = True
                dataClass[i, :] = minIndex, minDistance**2
        
        pointsCollection = []
        #对每一个类别进行搜索
        for j in range(k):
           pointsCollection = data[nonzero(dataClass[:, 0] == j)[0]]
           print("pointsCollection ：")
           print(pointsCollection)
           centerpoint[j,:] = mean(pointsCollection,axis = 0)

        return centerpoint, dataClass



data = [[1,1],[2,1],[1,2],[2,2],[4,3],[5,3],[4,4],[5,4]]
data = np.array(data)
centerpoint, dataClass = kmeans(data,2)
print("簇中心:")
print(centerpoint)
print("所属簇:")
print(dataClass)


