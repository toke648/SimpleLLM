import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']

# def house_data():
#     # 设置随机种子确保可重复
#     np.random.seed(0)
#     # 房屋大小（50到150平方米之间）
#     size = np.random.uniform(50, 150, 50)
#     # 根据线性关系计算价格（单位：万元）
#     price = 5 * size + 50 + np.random.normal(0, 10, 50)  # 加上一些噪声
#     # 创建DataFrame
#     data = pd.DataFrame({'房屋大小(㎡)': size, '价格(万元)': price})
#     data.to_excel("房屋价格数据.xlsx", index=False)
#     # 显示数据
#
#
# if __name__ == '__main__':
#
#     dataset = pd.read_excel('房屋价格数据.xlsx')
#
#     x = np.array(dataset['房屋大小(㎡)'])
#     y = np.array(dataset['价格(万元)'])
#
#     plt.figure(figsize=(8, 6))
#     plt.scatter(x, y, color='b', label='房屋大小 vs 价格')
#
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.grid(True)
#
#     plt.legend()
#
#     plt.show()


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # 设置随机种子确保可重复
#
# # 房屋大小（50到150平方米之间）
# size = np.random.uniform(50, 150, 50)
#
# # 根据线性关系计算价格（单位：万元）
# price = 5 * size + 50 + np.random.normal(0, 10, 50)  # 加上一些噪声
#
# # 创建DataFrame
# data = pd.DataFrame({'房屋大小(㎡)': size, '价格(万元)': price})
#
# # 绘制散点图
# plt.figure(figsize=(8, 6))
# plt.scatter(data['房屋大小(㎡)'], data['价格(万元)'], color='b', label='房屋大小 vs 价格')
#
# # 设置图表标题和标签
# plt.title('房屋大小与价格关系散点图')
# plt.xlabel('房屋大小 (㎡)')
# plt.ylabel('价格 (万元)')
# plt.grid(True)
#
# # 显示图例
# plt.legend()
#
# # 显示图表
# plt.show()



# # 读取数据
# dataset = pd.read_excel('房屋价格数据.xlsx')  # 需要确保数据的路径正确
#
# # 获取数据
# x = np.array(dataset['房屋大小(㎡)'])
# y = np.array(dataset['价格(万元)'])
#
# # 将数据切分为训练集和测试集
# X_train = x[0:30]  # 训练集
# Y_train = y[0:30]  # 训练集标签
# X_test = x[30:]  # 测试集
# Y_test = y[30:]  # 测试集标签
# n_train = len(X_train)
#
# # 初始化参数
# w = 0.0  # 权重初始化为0
# b = 0.0  # 截距初始化为0
# lr = 0.001  # 学习率
# epoch = 5000  # 迭代次数
#
#
# # 线性回归模型
# def model(x):
#     return w * x + b
#
#
# # 损失函数：均方误差
# def compute_loss(X, Y):
#     m = len(X)
#     predictions = model(X)
#     loss = np.sum((predictions - Y) ** 2) / m  # 均方误差
#     return loss
#
#
# # 训练模型：梯度下降
# def train(X, Y, epochs, learning_rate):
#     global w, b
#     loss_history = []
#
#     for epoch in range(epochs):
#         # 计算预测值
#         predictions = model(X)
#
#         # 计算误差
#         error = predictions - Y
#
#         # 计算梯度
#         gradient_w = np.sum(error * X) / n_train
#         gradient_b = np.sum(error) / n_train
#
#         # 更新参数
#         w -= learning_rate * gradient_w
#         b -= learning_rate * gradient_b
#
#         # 计算损失
#         loss = compute_loss(X, Y)
#         print(loss)
#         loss_history.append(loss)
#
#         # 每1000次输出一次损失值
#         if epoch % 1000 == 0:
#             print(f"Epoch {epoch}, Loss: {loss}")
#
#     return loss_history
#
#
# # 训练模型
# loss_history = train(X_train, Y_train, epoch, lr)
#
# # 绘制损失曲线
# plt.plot(range(epoch), loss_history)
# plt.title('训练过程中的损失值变化')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.show()
#
#
# # 绘制回归直线
# def plot_regression_line(X, Y):
#     plt.figure(figsize=(8, 6))
#     plt.scatter(X, Y, color='blue', label='实际数据')
#     plt.plot(X, model(X), color='red', label='回归直线')
#     plt.xlabel('房屋大小 (㎡)')
#     plt.ylabel('价格 (万元)')
#     plt.title('线性回归模型')
#     plt.legend()
#     plt.show()
#
#
# # 绘制回归结果
# plot_regression_line(x, y)
#
#
# # 测试模型性能
# def test_model(X_test, Y_test):
#     predictions = model(X_test)
#     mse = np.mean((predictions - Y_test) ** 2)
#     print(f"测试集的均方误差: {mse}")
#
#
# # 测试模型
# test_model(X_test, Y_test)


import numpy as np
import matplotlib.pyplot as plt

# 生成随机数据
np.random.seed(42)  # 保证结果可复现
size = np.random.uniform(50, 150, 50)  # 样本的 x 值
price = 1 * size + 50 + np.random.normal(0, 10, 50)  # y = 1*x + 50 + 噪声

# 数据归一化
size = (size - np.min(size)) / (np.max(size) - np.min(size))
price = (price - np.min(price)) / (np.max(price) - np.min(price))

# 可视化数据
plt.scatter(size, price, color='red', label='Data points')
plt.title('Scatter plot of size vs price')
plt.xlabel('Size (x)')
plt.ylabel('Price (y)')
plt.legend()
plt.show()


# 线性回归的实现
class LinearRegression:
    def __init__(self, learning_rate, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = np.random.randn() * 0.01  # 初始化 w
        self.b = np.random.randn() * 0.01  # 初始化 b

    def predict(self, x):
        """ 预测函数： y = wx + b """
        return self.w * x + self.b

    def compute_loss(self, y_true, y_pred):
        """ 计算 MSE 损失函数 """
        return np.mean((y_true - y_pred) ** 2)

    def train(self, x, y):
        """ 使用梯度下降训练模型 """
        n = len(x)
        loss_history = []

        for epoch in range(self.epochs):
            # 预测
            y_pred = self.predict(x)

            # 计算损失
            loss = self.compute_loss(y, y_pred)
            loss_history.append(loss)

            # 计算梯度
            dw = -(2 / n) * np.sum((y - y_pred) * x)
            db = -(2 / n) * np.sum(y - y_pred)

            # 梯度裁剪
            dw = np.clip(dw, -1, 1)
            db = np.clip(db, -1, 1)

            # 更新权重和偏置
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            # 每 100 个 epoch 打印一次损失
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss:.4f}, w: {self.w:.4f}, b: {self.b:.4f}')

        return loss_history

    def plot_loss(self, loss_history):
        """ 绘制损失随 epoch 的变化曲线 """
        plt.plot(range(len(loss_history)), loss_history, label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss during training')
        plt.legend()
        plt.show()


# 训练模型
model = LinearRegression(learning_rate=0.01, epochs=1000)
loss_history = model.train(size, price)
model.plot_loss(loss_history)

# 可视化回归线
plt.scatter(size, price, color='red', label='Data points')
x_range = np.linspace(min(size), max(size), 100)
y_range = model.predict(x_range)
plt.plot(x_range, y_range, color='blue', label='Regression Line')
plt.title('Linear Regression Fit')
plt.xlabel('Size (x)')
plt.ylabel('Price (y)')
plt.legend()
plt.show()




