import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math

# 设置字体为黑体
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

def func(x):
    y = x ** 2 + 1
    # y = math.exp(1) ** x
    return y

# 计算函数的梯度
def get_gradient(x):
    gradient = 2 * x
    return gradient

# 利用梯度下降算法找到使得函数取得最小值的自变量的值
def get_x_star(xi):
    for i in range(epochs):
        trajectory.append(xi)  # 存储当前的 xi 值
        xi = xi - lr * get_gradient(xi)  # 使用梯度下降更新 xi
    x_star = xi
    return x_star

if __name__ == '__main__':
    # 指定迭代次数
    epochs = 50
    # 指定学习率
    lr = 0.1
    # 初始化自变量
    xi = -18

    # 存储更新后的值
    trajectory = []

    # 打印出使函数最小化的 x 值
    print("x_star:", get_x_star(xi))

    # 生成用于绘制函数的数据
    x = np.arange(-20, 20, 0.1)
    y = func(x)

    # 绘制函数图像
    plt.plot(x, y, label='y = x^2 + 1')

    # 将 trajectory 转换为 NumPy 数组
    trajectory = np.array(trajectory)
    y_trajectory = func(trajectory)

    # 绘制散点图表示轨迹点
    plt.scatter(trajectory, y_trajectory, color='red', label='梯度下降轨迹')
    plt.title('在 $y = x^2 + 1$ 上的梯度下降')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()


