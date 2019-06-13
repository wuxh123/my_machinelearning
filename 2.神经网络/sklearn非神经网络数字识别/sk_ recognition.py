from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
import numpy as np
import scipy
import cv2
from fractions import Fraction
import imageio
import skimage

	
def image2Digit(image):
    # 调整为8*8大小
    im_resized = scipy.misc.imresize(image, (8,8))
    # RGB（三维）转为灰度图（一维）
    im_gray = cv2.cvtColor(im_resized, cv2.COLOR_BGR2GRAY)
    # 调整为0-16之间（digits训练数据的特征规格）像素值——16/255
    im_hex = Fraction(16,255) * im_gray
    # 将图片数据反相（digits训练数据的特征规格——黑底白字）
    im_reverse = 16 - im_hex
    return im_reverse.astype(np.int)
# 加载数字数据
digits = load_digits()
#训练集
X = digits.data
#标记
Y = digits.target
 
#数据与处理，让特征值都处在0-1之间
X -= X.min()
X /= X.max()

# 划分训练集与验证集
Xtrain, Xtest, ytrain, ytest = train_test_split(X, Y, test_size=1,random_state=1)
# 创建模型
clf = LogisticRegression(penalty='l2')
# 拟合数据训练
clf.fit(Xtrain, ytrain)
# 预测验证集
ypred = clf.predict(Xtest)
# 计算准确度
accuracy = accuracy_score(ytest, ypred)
print("识别准确度：",accuracy)
 
# 读取单张自定义手写数字的图片
image = imageio.imread("digit_image/8.png")
# 将图片转为digits训练数据的规格——即数据的表征方式要统一
im_reverse = image2Digit(image)
# 显示图片转换后的像素值
print(im_reverse)
# 8*8转为1*64（预测方法的参数要求）
reshaped = im_reverse.reshape(1,64)
# 预测
result = clf.predict(reshaped)
print("result=",result)