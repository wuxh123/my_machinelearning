import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
#准备数据集
iris=datasets.load_iris()
X=iris.data
print('X:\n',X)
Y=iris.target
print('Y:\n',Y)
 
#处理二分类问题，所以只针对Y=0,1的行，然后从这些行中取X的前两列
x=X[Y<2,:2]
print(x.shape)
print('x:\n',x)
y=Y[Y<2]
print('y:\n',y)
#target=0的点标红，target=1的点标蓝,点的横坐标为data的第一列，点的纵坐标为data的第二列
plt.scatter(x[y==0,0],x[y==0,1],color='red')
plt.scatter(x[y==1,0],x[y==1,1],color='green')

plt.scatter(5.6,3.2,color='blue')
x_1=np.array([5.6,3.2])
#plt.title('红色点标签为0,绿色点标签为1，待预测的点为蓝色')
#plt.show()

#采用欧式距离计算
distances=[np.sqrt(np.sum((x_t-x_1)**2)) for x_t in x]
#对数组进行排序，返回的是排序后的索引
d=np.sort(distances)
nearest=np.argsort(distances)
k=6
topk_y=[y[i] for i in nearest[:k]]
from collections import Counter
#对topk_y进行统计返回字典
votes=Counter(topk_y)
#返回票数最多的1类元素
print(votes)
predict_y=votes.most_common(1)[0][0]
print(predict_y)
plt.show()
