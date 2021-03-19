from sklearn import preprocessing
import numpy as np

'ont-hot'

enc = preprocessing.OneHotEncoder()  # 通过索引
enc1 = preprocessing.OneHotEncoder(sparse=False)

ans = enc.fit_transform([[0], [1], [2], [1]])
ans1 = enc1.fit_transform([[3], [1], [2], [1]])
print(ans)
print(ans1)
