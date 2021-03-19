import numpy as np
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

rng = np.random.RandomState(0)
# rng = np.random

# 核岭回归
X = 5 * rng.rand(100, 1)
y = np.sin(X).ravel()
# print(np.sin(X))
# print(y)

# Add noise to targets
# y[::5] += 3 * (0.5 - rng.rand(X.shape[0] // 5))#20
y[::5] += 3 * (0.5 - rng.rand(20, 1).ravel())
# print(y)
# print(y[::5])
# kr = KernelRidge(kernel='sigmoid', alpha=0.3,gamma=0.3)
# kr = KernelRidge(kernel='linear', alpha=0.5,gamma=0.5)
# kr = KernelRidge(kernel='rbf', alpha=0.5,gamma=0.5)
print(np.logspace(-2, 2, 5))
print([1e-02, 1e-01, 1e+00, 1e+01, 1e+02])

kr = GridSearchCV(KernelRidge(),  # 表格搜索
                  param_grid={"kernel": ["rbf", "laplacian", "polynomial", "sigmoid"],
                              "alpha": [1e0, 0.1, 1e-2, 1e-3],
                              "gamma": np.logspace(-2, 2, 5)})
kr.fit(X, y)
print(kr.best_score_, kr.best_params_)

X_plot = np.linspace(0, 5, 100)
y_kr = kr.predict(X_plot[:, None])
# y_kr = kr.predict(np.expand_dims(X_plot,1))

plt.scatter(X, y)
plt.plot(X_plot, y_kr, color="red")
plt.show()
