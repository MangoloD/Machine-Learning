from sklearn import datasets, preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score  # 引入交叉验证
import matplotlib.pyplot as plt

# 引入数据
iris = datasets.load_iris()
X = iris.data
Y = iris.target
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 设置n_neighbors的值为1到30,通过绘图来看训练分数###
k_range = range(1, 31)
k_score = []
t_score = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    scores = cross_val_score(knn, x_train, y_train, cv=5, scoring='accuracy')  # for classfication
    y_pred = knn.predict(x_test)
    scores_t = accuracy_score(y_test, y_pred)
    k_score.append(scores.mean())
    t_score.append(scores_t)
plt.figure()
plt.plot(k_range, k_score, color='green', label='train')
plt.plot(k_range, t_score, color='red', label='test')
plt.xlabel('Value of k for KNN')
plt.ylabel('CrossValidation accuracy')
plt.legend()
plt.show()
