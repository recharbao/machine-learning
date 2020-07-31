from train.train import train
from extractData.extract import extract_data
from extractData.extract import extract_labels
import numpy as np
from tqdm import tqdm
from predict.predict import predict


 #训练数据
params = train()
[f1, f2, w3, w4, b1, b2, b3, b4] = params

#测试数据数目
m =10000
X = extract_data('F:\handwrite\\t10k-images.idx3-ubyte', m, 28)
y_label = extract_labels('F:\handwrite\\t10k-labels.idx1-ubyte', m).reshape(m,1)
#数据预处理
X-= int(np.mean(X)) # subtract mean
X/= int(np.std(X)) # Normalize
test_data = np.hstack((X,y_label)) #绑定

X = test_data[:,0:-1]
X = X.reshape(len(test_data), 1, 28, 28)
y = test_data[:,-1]

#预测正确的个数
corr = 0

print()
print("计算对测试集的预测准确率:")

#进度条
t = tqdm(range(len(X)), leave=True)

for i in t:
    x = X[i]
    pred, prob = predict(x, f1, f2, w3, w4, b1, b2, b3, b4)
    if pred==y[i]:
        corr+=1
    t.set_description("Acc:%0.2f%%" % (float(corr/(i+1))*100))
    
print("Overall Accuracy: %.2f" % (float(corr/len(test_data)*100)))
