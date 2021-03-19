import numpy as np
import matplotlib.pyplot as plt


def normalization0(x):
    """ 归一化（0~1）"""
    '''x_=(x−x_min)/(x_max−x_min)'''
    return [(float(i) - min(x)) / float(max(x) - min(x)) for i in x]


def normalization1(x):
    """最值归一化（-1~1）"""
    '''x_=(x)/(x_max)
           ....                       '''
    x_ = [float(i) / np.max(np.abs(x)) for i in x]  # [-1,1)||(-1,1]
    # x_ = [(float(i)-np.mean(x))/np.max(np.abs(x-np.mean(x))) for i in x]#[-1,1)||(-1,1]
    return x_


def normalization2(x):
    """均值归一化（-1~1）"""
    '''x_=(x−x_mean)/(x_max−x_min)'''
    '''x_=(2*x−x_max−x_min)/(x_max−x_min)'''
    # return [(float(i)-np.mean(x))/(max(x)-min(x)) for i in x]
    return [(float(2 * i) - np.max(x) - np.min(x)) / (max(x) - min(x)) for i in x]  # [-1,1]


def normalization3(x):
    """标准化（μ=0，σ=1）"""
    '''x =(x−μ)/σ'''
    x_mean = np.mean(x)
    s2 = np.mean([(i - np.mean(x)) ** 2 for i in x])
    std = np.sqrt(s2)

    return [(i - x_mean) / (s2 + 1e-10) for i in x]  # x减均值，除以方差
    # return [(i-x_mean)/(std+0.00001) for i in x]#x减均值，除以标准差


def normalization4(x):
    """归一化：只有全是非负数的情况下使用,[-1,1]"""
    '''x =((x/x_max)-0.5)/0.5'''
    x_mean = [(float(i) / np.max(x) - 0.5) / 0.5 for i in x]
    return x_mean


s = [21312313, 0, 76, 223, 12, -1341656]
l_ = [3, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11,
      12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
cs = []
for i in l_:
    c = l_.count(i)
    cs.append(c)
print(cs)

n0 = normalization0(l_)
n1 = normalization1(l_)
n2 = normalization2(l_)
n3 = normalization3(l_)
n4 = normalization4(l_)

print(n0)
print(n1)
print(n2)
# print(n3)
# plt.plot(l,cs)
plt.plot(n0, cs, c="r")
plt.plot(n1, cs, c="blue")
# plt.plot(n2,cs,c="g")
plt.plot(n3, cs, c="y")
plt.plot(n4, cs, c="k")

plt.show()
