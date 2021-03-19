from sklearn import preprocessing
import numpy as np

'1.preprocessing.scale(X),可以直接将给定数据进行标准化。'
# 将每一列的特征标准化，每一列表示同一类特征，类似于图片每个通道上的对应点
X = np.array([[1., -1., 2.],
              [2., 0., 0.],
              [0., 1., -1.]])
X_scaled = preprocessing.scale(X)
print(X_scaled)

"""
array([[0...., -1.22..., 1.33...],
       [1.22..., 0...., -0.26...],
       [-1.22..., 1.22..., -1.06...]])"""

# 处理后数据的均值和方差
print(X_scaled.mean(axis=0))
"array([0., 0., 0.])"

print(X_scaled.std(axis=0))
"array([1., 1., 1.])"

'2.preprocessing.StandardScaler().fit(X),保存训练集中的参数（均值、方差）,直接使用其对象标准化转换测试集数据。'
scaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True).fit(X)
print(scaler)

print(scaler.transform(X))
"""array([[0...., -1.22..., 1.33...],
       [1.22..., 0...., -0.26...],
       [-1.22..., 1.22..., -1.06...]])"""

# 可以直接使用训练集对测试集数据进行转换
print(scaler.transform([[-1., 1., 0.]]))
"array([[-2.44..., 1.22..., -0.26...]])"

print(scaler.mean_)
print(scaler.var_)
print(np.array([1., 2, 0]).mean())
print(np.array([1., 2, 0]).var())
print(np.array([-1., 0, 1]).mean())
print(np.array([-1., 0, 1]).var())
print(np.array([2., 0, -1]).mean())
print(np.array([2., 0, -1]).var())
'''
[[1., -1., 2.],
  [2., 0., 0.],
  [0., 1., -1.]]
'''
