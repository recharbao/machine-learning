
def predict(image, f1, f2, w3, w4, b1, b2, b3, b4, conv_step = 1, pool_dim = 2, pool_step = 2):
   
    conv1 = convolution(image, f1, b1, conv_step) # 卷积操作
    conv1[conv1<=0] = 0 #ReLU
    
    conv2 = convolution(conv1, f2, b2, conv_step) # 卷积操作
    conv2[conv2<=0] = 0 # ReLU
    
    pooled = maxpool(conv2, pool_dim, pool_step) # 池化
    (pool_depth, dim2, _) = pooled.shape
    fc = pooled.reshape((pool_depth * dim2 * dim2, 1)) # 全连接
    
    z = w3.dot(fc) + b3 # 第一个隐藏层
    z[z<=0] = 0 #ReLU 
    
    out = w4.dot(z) + b4 # 第二个隐藏层
    probs = softmax(out) # 预测
    
    return np.argmax(probs), np.max(probs)
