import numpy as np
def convolutionBackward(dconv_prev, conv_in, filt, step):
 
    (num, depth_filt, dim_filt, _) = filt.shape
    (_, orig_dim, _) = conv_in.shape
    ## initialize derivatives
    dout = np.zeros(conv_in.shape)
    dfilt = np.zeros(filt.shape)
    dbias = np.zeros((num,1))
    for curr_filt in range(num):
        # 循环卷积核
        curr_y = out_y = 0
        while curr_y + dim_filt <= orig_dim:
            curr_x = out_x = 0
            while curr_x + dim_filt <= orig_dim:
                # 卷积核的损失梯度
                dfilt[curr_filt] += dconv_prev[curr_filt, out_y, out_x] * conv_in[:, curr_y:curr_y+dim_filt, curr_x:curr_x+dim_filt]
                # 上一层的delta
                dout[:, curr_y:curr_y+dim_filt, curr_x:curr_x+dim_filt] += dconv_prev[curr_filt, out_y, out_x] * filt[curr_filt] 
                curr_x += step
                out_x += 1
            curr_y += step
            out_y += 1
        #  bias的损失梯度
        dbias[curr_filt] = np.sum(dconv_prev[curr_filt])
    
    return dout, dfilt, dbias

