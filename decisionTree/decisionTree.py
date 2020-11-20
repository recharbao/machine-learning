#-*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import numpy as np
import pandas as pd
import math

#ID3算法
class ID3(object):
    def __init__(self):
        self.data = None
        self.len = None
    
    def CreateData(self):

        dataC = np.array([['long', 'thick', 'no', 'man'],
                        ['short', 'medium', 'no', 'man'],
                        ['short', 'thin', 'yes', 'man'],
                        ['short', 'thick', 'no', 'man'],
                        ['long', 'medium', 'no', 'man'],
                        ['short', 'thick', 'no', 'man'],
                        ['long', 'thick', 'yes', 'man'],
                        ['short', 'thin', 'no', 'man'],
                        ['long', 'thin', 'yes', 'woman'],
                        ['long', 'medium', 'yes', 'woman'],
                        ['long', 'thick', 'yes', 'woman'],
                        ['short', 'thin', 'yes', 'woman'],
                        ['long', 'thin', 'no', 'woman'],
                        ['short', 'medium', 'no', 'woman'],
                        ['long', 'thick', 'yes', 'woman'],
                        ['long', 'thin', 'no', 'woman'],
                        ['short', 'medium', 'yes', 'woman']])

        self.len = len(dataC)

        feature = np.array(['hair', 'voice', 'ear_stud', 'labels'])
        self.data = pd.DataFrame(dataC, columns=feature)
        return self.data

    def get_Ent(self, data):

        num_data = len(data)
        countClass = {}

        for i in range(num_data):
            new_data = data.iloc[i, :]
            label = new_data['labels']

            if label not in countClass.keys():
                countClass[label] = 0
            countClass[label] += 1

        Ent = 0.0
        for key in countClass:
            prob = float(countClass[key]) / num_data
            Ent -= prob * math.log(prob, 2)

        return Ent


    def get_gain(self, data, origin_Ent, feature):

        featureValue = data[feature]
        unique_featureValue = set(featureValue)
        Ent = 0.0

        for fea in unique_featureValue:
            new_data = data[data[feature] == fea]
            w = float(len(new_data)) / len(featureValue)
            Ent += w * self.get_Ent(new_data)

        return origin_Ent - Ent


    def choose_best(self, data):
        
        num_feature = len(data.columns) - 1 #特征的数量
        origin_Ent = self.get_Ent(data)
        best_feature = data.columns[0] 
        best_gain = 0

        for i in range(num_feature):
            #计算信息增益
            new_gain = self.get_gain(data, origin_Ent, data.columns[i])
            if new_gain > best_gain:
                new_gain = best_gain
                best_feature = data.columns[i]

        return best_feature


    def create_tree(self, data):
        
        feature = data.columns[:-1].tolist()
        label = data.iloc[:, -1]
        
        #完全归为一类
        if len(data['labels'].value_counts()) == 1:
            node = data['labels'].mode().values
            return node

        #如果说类别分完
        if len(feature) == 1:
            node = data['labels'].mode().values
            return node
        

        #选择一个最优特征作为根
        best_feature = self.choose_best(data)

        tree = {best_feature:{}}

        best_feature_value = data[best_feature]

        unique_value = set(best_feature_value)

        for each in unique_value:
            datax = data[data[best_feature] == each]
            datax = datax.drop([best_feature], axis=1)
            tree[best_feature][each] = self.create_tree(datax)

        return tree

    def predict(self, tree, test):
        first_feature = list(tree.keys())[0]  # 获取根节点
        feature_dict = tree[first_feature]  # 根节点下的树
        labels = test.columns.tolist()
        value = test[first_feature][0]
        for key in feature_dict.keys():
            if value == key:
                if type(feature_dict[key]).__name__ == 'dict':  # 判断该节点是否为叶节点
                    class_label = self.predict(feature_dict[key], test)  # 采用递归直到遍历到叶节点
                else:
                    class_label = feature_dict[key]
        return class_label
    


if __name__ == "__main__":
    
    id3 = ID3()
    data = id3.CreateData()
    #print(data.iloc[1])
    tree = id3.create_tree(data)
    #print(tree)
    test = pd.DataFrame({"hair": ["long"], "voice": ["thin"], "ear_stud": ["yes"]})
    pred = id3.predict(tree, test)
    print(pred)


