
import numpy as np

#计算损失
#注意此处使用的是:categorical cross-entropy loss
def calculateLoss(probs,label):
    return -np.sum(label * np.log(probs))