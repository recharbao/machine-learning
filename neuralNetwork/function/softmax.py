import numpy as np

#至于为什么使用softmax,该文件夹中有图片解释，更详细的解释参照下面:
#1?https://zhuanlan.zhihu.com/p/45014864
#2?http://freemind.pluskid.org/machine-learning/softmax-vs-softmax-loss-numerical-stability/

def softmax(raw_preds):
    out = np.exp(raw_preds)
    return out/np.sum(out)