import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 生成 x 和真实 y 值
epochs = 20
x = np.arange(epochs)
y_start = x + 1  # 初始的目标函数 y = x + 1
y_true = 2 * x + 1  # 目标逼近的函数 y = 2x + 1

# 初始化拟合的参数 a 和 b
a = 0.0
b = 0.0

# 学习率
lr = 0.1


# 计算均方误差损失
def compute_loss(a, b, x, y_true):
    y_pred = a * x + b
    return np.mean((y_pred - y_true) ** 2)


# 计算梯度
def compute_gradients(a, b, x, y_true):
    y_pred = a * x + b
    grad_a = -2 * np.mean((y_true - y_pred) * x)
    grad_b = -2 * np.mean(y_true - y_pred)
    return grad_a, grad_b


# 更新参数
def update_parameters(a, b, grad_a, grad_b, lr):
    a -= lr * grad_a
    b -= lr * grad_b
    return a, b


# 创建图形
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(0, epochs - 1)
ax.set_ylim(0, max(y_true) + 1)
ax.set_title('Linear Regression Fitting from y = x + 1 to y = 2x + 1')
ax.set_xlabel('x')
ax.set_ylabel('y')

# 绘制初始目标函数 y = x + 1
line_start, = ax.plot(x, y_start, label='Starting Function (x + 1)', color='green', linestyle='--')

# 绘制目标函数 y = 2x + 1
line_true, = ax.plot(x, y_true, label='True Function (2x + 1)', color='red')

# 拟合的蓝色直线
line_fit, = ax.plot([], [], label='Fitted Line', color='blue')

# 显示损失函数
loss_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')


# 更新动画
def update(frame):
    global a, b
    # 计算梯度
    grad_a, grad_b = compute_gradients(a, b, x, y_start)
    # 更新参数
    a, b = update_parameters(a, b, grad_a, grad_b, lr)

    # 计算拟合的直线
    y_fit = a * x + b
    line_fit.set_data(x, y_fit)

    # 计算并显示损失
    loss = compute_loss(a, b, x, y_true)
    loss_text.set_text(f'Loss: {loss:.4f}')

    return line_fit, loss_text


# 创建动画
ani = FuncAnimation(fig, update, frames=100, interval=200, blit=True)

# 显示图例
ax.legend()

# 展示动画
plt.show()

#
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 定义生成 x 值的函数
# def func(epochs):
#     # 返回从 0 到 epochs - 1 的数组
#     return np.arange(epochs)
#
# epochs = 20
# # 学习率
# lr = 0.1
#
# # 生成 x 值，调用 func
# x = func(epochs)
#
# # 生成 y 值，可以选择用一个简单的线性关系，比如 y = 2x
# y1 = 2 * x
#
# # 绘图
# plt.plot(x, y1, label='y = 2x', color='blue')
#
# # 如果你想表示学习率影响下的变化，可以使用另外一个公式
# # 这里我们用一个假设的公式 y = x + lr * x
# y2 = x + lr * x  # 这里的学习率影响可以简单理解为对 x 的线性放缩
#
# plt.plot(x, y2, label='y = x + lr * x', color='red')
#
# # 添加标题和标签
# plt.title('Plots of y = 2x and y = x + lr * x')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.grid(True)
#
# # 显示图形
# plt.show()




