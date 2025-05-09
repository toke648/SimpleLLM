# class linear():
#     def __init__(self):
#         pass
#
#     def func(self, k: int, x: int, b: int) -> None:
#         y = k * x + b
#         print()

import numpy as np
import matplotlib.pyplot as plt

def func(x):
    y = x ** 2 + 1
    return y

# 指定自变量代送次数
epochs = 50
# 指定学习率
lr = 0.1
# 对自变量的值进行初始化
xi = -18
# 求函数的梯度值
def get_gradient(x):
    gradient = 2 * x
    return gradient

# 存储更新后的值
trajectory = []

# 利用梯度下降算法找到使得函数取得最小值的自变量的值x_star
def get_x_star(xi):
    for i in range(epochs):
        trajectory.append(xi)
        xi = xi - lr * get_gradient(xi)
    x_star = xi
    return x_star

print(get_x_star(xi))

x = np.arange(-20, 20, 0.1)
y = func(x)
plt.plot(x, y)

x_trajectory = np.arange(trajectory)
y_trajectory = func(trajectory)

plt.scatter(x_trajectory, y_trajectory)
plt.show()

# def forward():
#     x =