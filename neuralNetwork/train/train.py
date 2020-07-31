from extractData.extract import extract_data
from extractData.extract import extract_labels
from initData.initData import initFilter
from initData.initData import initWeight
import numpy as np
from tqdm import tqdm
from optimize.adamGD import adamGD

def train(num_classes = 10,lr = 0.01,beta1 = 0.95,beta2 = 0.99,data_dim = 28,data_depth = 1,f = 5 ,num_filt1 = 8,num_filt2 = 8,batch_size = 32,num_epochs = 2):

    m = 50000
    X = extract_data('F:\handwrite\\train-images.idx3-ubyte',m,data_dim)
    y_label = extract_labels('F:\handwrite\\train-labels.idx1-ubyte',m).reshape(m,1)

    X -= int(np.mean(X))  #subtract mean
    X /= int (np.std(X))  # Normalize
    train_data = np.hstack((X,y_label))

    np.random.shuffle(train_data) #打乱数据

    f1,f2,w3,w4 = (num_filt1,data_depth,f,f),(num_filt2,num_filt1,f,f),(128,800),(10,128)
    #初始化卷积核，权值矩阵
    f1 = initFilter(f1)
    f2 = initFilter(f2)
    w3 = initWeight(w3)
    w4 = initWeight(w4)


    b1 = np.zeros((f1.shape[0],1))
    b2 = np.zeros((f2.shape[0],1))
    b3 = np.zeros((w3.shape[0],1))
    b4 = np.zeros((w4.shape[0],1))


    params = [f1,f2,w3,w4,b1,b2,b3,b4]

    cost = []

    print("LearnRate:"+str(lr)+",Batch_Size:"+str(batch_size))

    for epoch in range(num_epochs):

        #打乱数据
        np.random.shuffle(train_data)
        batches = [train_data[k:k + batch_size] for k in range(0,train_data.shape[0],batch_size)]

        t = tqdm(batches)
        for x,batch in enumerate(t):
            params,cost = adamGD(batch,num_classes,lr, data_dim, data_depth, beta1, beta2, params, cost)
            t.set_description("Cost: %.5f" % (cost[-1]))

    return params

