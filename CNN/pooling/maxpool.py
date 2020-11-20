import numpy as np

#池化
#通过池化可以保留主要特征，实现数据的降维，进而减小计算量
#从直觉出发其实也可以感知，实在不行就画实例图感受一下
def maxpool(data, dim=2, step=2):
    depth_data, h_prev, w_prev = data.shape
    
    # 计算输出维度
    h = int((h_prev - dim)/step)+1 
    w = int((w_prev - dim)/step)+1
    
    
    downsampled = np.zeros((depth_data, h, w)) 
    
    # 对每一层数据进行池化
    for i in range(depth_data):
        curr_y = out_y = 0
        # 移动窗口
        while curr_y + dim <= h_prev:
            curr_x = out_x = 0
            while curr_x + dim <= w_prev:
                # 最大池化
                downsampled[i, out_y, out_x] = np.max(data[i, curr_y:curr_y+dim, curr_x:curr_x+dim])
                curr_x += step
                out_x += 1
            curr_y += step
            out_y += 1
    return downsampled


