# 加载依赖库
import numpy as np
import matplotlib.pyplot as plt

# # 定义 y=x^2+1 函数
# def function(x):
#     y = x ** 2 + 1
#     return y
#
# # 指定自变量更新迭代次数（迭代的次数）
# epoch = 50
# # 指定学习率
# lr = 0.1
# # 初始化变量
# xi = -10
#
# # 求函数梯度
# def get_gradient(x):
#     gradient = 2 * x
#     return gradient
#
# # 储存更新后的梯度值
# trajectory = []
#
# # 利用梯度下降算法找到使函数取得最小值的自变量的值x_star
# def get_x_star(xi):
#     for i in range(epoch):
#         trajectory.append(xi)
#         print(xi)
#         xi = xi - lr * get_gradient(xi)
#     x_star = xi
#     return x_star
#
# get_x_star(xi)
# # print(trajectory)
#
# x = np.arange(-10, 10, 0.1)
# y = function(x)
#
# # 画出函数图像
# plt.plot(x, y)
# x_trajectory = np.array(trajectory)
# y_trajectory = function(x_trajectory)
#
# # Scatter plot for the trajectory
# plt.scatter(x_trajectory, y_trajectory, color='red', label='Gradient Descent Steps')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Gradient Descent on $f(x) = x^2 + 1$')
# plt.legend()
# plt.grid()
# plt.show()




if __name__ == '__main__':
    epoch = 50
    lr = 0.1
    xi = -18

    trajectory = []

    for i in range(epoch):
        trajectory.append(xi)
        print(xi)
        xi = xi - lr * function()

    x = np.arange(-20, 20, 0.1)
    y = x ** 2 + 1

    # 画出函数图像
    plt.plot(x, y)
    x_trajectory = np.array(trajectory)
    y_trajectory = x_trajectory ** 2 + 1

    plt.scatter(x_trajectory, y_trajectory, color="red", label='Gradient Descent Steps')
    plt.title('Gradient Descent on $f(x) = x^2 + 1$')
    plt.xlabel("x")
    plt.ylabel('y')
    # tap标签
    plt.legend()
    # grid 网格背景
    plt.grid()
    plt.show()

import torch

print(torch.arange(12))


