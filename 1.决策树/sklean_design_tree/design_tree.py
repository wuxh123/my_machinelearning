from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import tree
from sklearn import preprocessing


# Read in the csv file and put features into list of dict and list of class label

allElectronicsData = open(r'AllElectronics.csv', 'rt')
reader = csv.reader(allElectronicsData)

headers = next(reader)    # 读取表格的第一行，即表头


featureList = []  # 特征值列表
labelList = []    # 标签列表，即class_buys_computer的内容


# 从表格的第二行开始，一行一行的循环整个cvs
for row in reader:
    labelList.append(row[len(row)-1])  # 表格最后一列class_buys_computer的内容
    rowDict = {}
    for i in range(1, len(row)-1):   # 不读取第一列RID
        rowDict[headers[i]] = row[i]
    featureList.append(rowDict)


# 将特征值列表中的内容转化成向量形式
vec = DictVectorizer()
dummyX = vec.fit_transform(featureList) .toarray()
print(dummyX)

print("dummyX: " + str(dummyX))
print(vec.get_feature_names())

print("labelList: " + str(labelList))

# 将标签类的内容转化成向量形式
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
print("dummyY: " + str(dummyY))

# Using decision tree for classification
# clf = tree.DecisionTreeClassifier()
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummyX, dummyY)
print("clf: " + str(clf))


# Visualize model
# 生成allElectronicInformationGainOri.dot文件
with open("allElectronicInformationGainOri.dot", 'w') as f:
    f = tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=f)


# 利用原来的数据生成新的数据进行预测
# 取第一行
oneRowX = dummyX[0, :]
print("oneRowX: " + str(oneRowX))

# 修改后生成新的数据
newRowX = oneRowX
newRowX[0] = 1
newRowX[2] = 0
print("newRowX: " + str(newRowX))


# 进行预测
predictedY = clf.predict([newRowX])
print("predictedY: " + str(predictedY))