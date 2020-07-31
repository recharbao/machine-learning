from convolution.convolution import convolution
import numpy as np
from pooling.maxpool import maxpool
from function.loss import calculateLoss
from function.softmax import softmax
from convNetworkBackPropagation.convBP.convolutionBackward import convolutionBackward
from convNetworkBackPropagation.poolBP.maxpoolBackward import maxpoolBackward

def buildConv(data, label, params, conv_step, pool_f, pool_step):
    
    [f1, f2, w3, w4, b1, b2, b3, b4] = params 

    ###向前传播###
    conv1 = convolution(data, f1, b1, conv_step) # 卷积
    conv1[conv1<=0] = 0 #ReLU
    
    conv2 = convolution(conv1, f2, b2, conv_step) # 卷积
    conv2[conv2<=0] = 0 # ReLU
    
    pooled = maxpool(conv2, pool_f, pool_step) #池化
    
    (nf2, dim2, _) = pooled.shape
    fc = pooled.reshape((nf2 * dim2 * dim2, 1)) # 全连接
    
    z = w3.dot(fc) + b3 # 进入隐藏层
    z[z<=0] = 0 #ReLU
    
    out = w4.dot(z) + b4 #进入隐藏层
     
    probs = softmax(out) #进入预测函数

    #损失
    loss = calculateLoss(probs, label) 

    ###反向传播###
    dout = probs - label # 隐藏层2损失梯度
    dw4 = dout.dot(z.T) # w4损失梯度
    db4 = np.sum(dout, axis = 1).reshape(b4.shape) # biases损失梯度
    
    dz = w4.T.dot(dout) #隐藏层1损失梯度
    dz[z<=0] = 0 # 反向传播ReLU
    dw3 = dz.dot(fc.T) #w3损失梯度
    db3 = np.sum(dz, axis = 1).reshape(b3.shape) #biases损失梯度
    
    dfc = w3.T.dot(dz) # fc损失梯度
    dpool = dfc.reshape(pooled.shape) 
    
    dconv2 = maxpoolBackward(dpool, conv2, pool_f, pool_step) # 池化层的反向传播
    dconv2[conv2<=0] = 0 # 反向传播 ReLU
    

    dconv1, df2, db2 = convolutionBackward(dconv2, conv1, f2, conv_step) # 卷积层的反向传播
    dconv1[conv1<=0] = 0 # 反向传播 ReLU

    ddata, df1, db1 = convolutionBackward(dconv1, data, f1, conv_step) # 卷积层的反向传播
    
    grads = [df1, df2, dw3, dw4, db1, db2, db3, db4]
    
    return grads, loss






def adamGD(batch, num_classes, lr, dim, depth_data, beta1, beta2, params, cost):
    '''
    update the parameters through Adam gradient descnet.
    '''
    [f1, f2, w3, w4, b1, b2, b3, b4] = params
    
    X = batch[:,0:-1] # 获取样本
    X = X.reshape(len(batch), depth_data, dim, dim)
    Y = batch[:,-1] # 获取样本标签
    
    cost_ = 0
    batch_size = len(batch)
    
    # 初始化梯度，动量，RMS
    df1 = np.zeros(f1.shape)
    df2 = np.zeros(f2.shape)
    dw3 = np.zeros(w3.shape)
    dw4 = np.zeros(w4.shape)
    db1 = np.zeros(b1.shape)
    db2 = np.zeros(b2.shape)
    db3 = np.zeros(b3.shape)
    db4 = np.zeros(b4.shape)
    
    v1 = np.zeros(f1.shape)
    v2 = np.zeros(f2.shape)
    v3 = np.zeros(w3.shape)
    v4 = np.zeros(w4.shape)
    bv1 = np.zeros(b1.shape)
    bv2 = np.zeros(b2.shape)
    bv3 = np.zeros(b3.shape)
    bv4 = np.zeros(b4.shape)
    
    s1 = np.zeros(f1.shape)
    s2 = np.zeros(f2.shape)
    s3 = np.zeros(w3.shape)
    s4 = np.zeros(w4.shape)
    bs1 = np.zeros(b1.shape)
    bs2 = np.zeros(b2.shape)
    bs3 = np.zeros(b3.shape)
    bs4 = np.zeros(b4.shape)
    
    for i in range(batch_size):
        
        x = X[i]
        y = np.eye(num_classes)[int(Y[i])].reshape(num_classes, 1)
        
        grads, loss = buildConv(x, y, params, 1, 2, 2)
        [df1_, df2_, dw3_, dw4_, db1_, db2_, db3_, db4_] = grads
        
        df1+=df1_
        db1+=db1_
        df2+=df2_
        db2+=db2_
        dw3+=dw3_
        db3+=db3_
        dw4+=dw4_
        db4+=db4_

        cost_+= loss

    # 更新参数
        
    v1 = beta1*v1 + (1-beta1)*df1/batch_size # 动量法
    s1 = beta2*s1 + (1-beta2)*(df1/batch_size)**2 # RMS
    f1 -= lr * v1/np.sqrt(s1+1e-7) # Adam
    
    bv1 = beta1*bv1 + (1-beta1)*db1/batch_size
    bs1 = beta2*bs1 + (1-beta2)*(db1/batch_size)**2
    b1 -= lr * bv1/np.sqrt(bs1+1e-7)
   
    v2 = beta1*v2 + (1-beta1)*df2/batch_size
    s2 = beta2*s2 + (1-beta2)*(df2/batch_size)**2
    f2 -= lr * v2/np.sqrt(s2+1e-7)
                       
    bv2 = beta1*bv2 + (1-beta1) * db2/batch_size
    bs2 = beta2*bs2 + (1-beta2)*(db2/batch_size)**2
    b2 -= lr * bv2/np.sqrt(bs2+1e-7)
    
    v3 = beta1*v3 + (1-beta1) * dw3/batch_size
    s3 = beta2*s3 + (1-beta2)*(dw3/batch_size)**2
    w3 -= lr * v3/np.sqrt(s3+1e-7)
    
    bv3 = beta1*bv3 + (1-beta1) * db3/batch_size
    bs3 = beta2*bs3 + (1-beta2)*(db3/batch_size)**2
    b3 -= lr * bv3/np.sqrt(bs3+1e-7)
    
    v4 = beta1*v4 + (1-beta1) * dw4/batch_size
    s4 = beta2*s4 + (1-beta2)*(dw4/batch_size)**2
    w4 -= lr * v4 / np.sqrt(s4+1e-7)
    
    bv4 = beta1*bv4 + (1-beta1)*db4/batch_size
    bs4 = beta2*bs4 + (1-beta2)*(db4/batch_size)**2
    b4 -= lr * bv4 / np.sqrt(bs4+1e-7)
    

    cost_ = cost_/batch_size
    cost.append(cost_)

    params = [f1, f2, w3, w4, b1, b2, b3, b4]
    
    return params, cost

