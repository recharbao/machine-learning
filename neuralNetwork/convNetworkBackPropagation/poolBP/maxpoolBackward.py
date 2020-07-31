import numpy as np
from function.nanargmax import nanargmax

def maxpoolBackward(dpool, orig, dim, step):
   
    (depth_orig, orig_dim, _) = orig.shape
    
    dout = np.zeros(orig.shape)
    
    for i in range(depth_orig):
        curr_y = out_y = 0
        while curr_y + dim <= orig_dim:
            curr_x = out_x = 0
            while curr_x + dim <= orig_dim:
                # 记录最大值的坐标
                (x, y) = nanargmax(orig[i, curr_y:curr_y+dim, curr_x:curr_x+dim])
                dout[i, curr_y+x, curr_x+y] = dpool[i, out_y, out_x]
                curr_x += step
                out_x += 1
            curr_y += step
            out_y += 1
        
    return dout