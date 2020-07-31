
#卷积
import numpy as np

def convolution(data, filt, bias, step=1):
    
    (num, depth_filt, dim_filt, _) = filt.shape # 卷积核的维度
    depth_data, dim_data, _ = data.shape # 数据的维度
    
    out_dim = int((dim_data - dim_filt)/step)+1 # 输出的维度
    
    
    assert depth_data ==depth_filt, "维度不匹配"
    
    out = np.zeros((num,out_dim,out_dim)) 
    
    # 用每一个卷积核卷积数据
    for curr_filt in range(num):
        curr_y = out_y = 0
        #移动卷积核
        while curr_y + dim_filt <= dim_data:
            curr_x = out_x = 0
            while curr_x + dim_filt <= dim_data:
                # 执行卷积
                out[curr_filt, out_y, out_x] = np.sum(filt[curr_filt] * data[:,curr_y:curr_y+dim_filt, curr_x:curr_x+dim_filt]) + bias[curr_filt]
                curr_x += step
                out_x += 1
            curr_y += step
            out_y += 1
        
    return out
