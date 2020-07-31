
import numpy as np

def initFilter(size,scale = 1.0):
    stddev = scale/np.sqrt(np.prod(size))
    return np.random.normal(loc=0,scale=stddev,size=size)

def initWeight(size):
    return np.random.standard_normal(size=size) * 0.01
