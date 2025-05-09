import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# 数据生成
T = 1000
time = torch.arange(1, T + 1, dtype=torch.float32)  # torch.Size([1000])

print(time)

# (T,)是一个元组，它只包含一个元素,表示生成一个形状为(T,)的一维张量
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))  # torch.Size([1000])

# 数据预处理
tau = 4
features = torch.zeros((T - tau, tau))  # torch.Size([996, 4])
for i in range(tau):
    features[:, i] = x[i: T - tau + i]  # features[:, i]是取第一列
labels = x[tau:].reshape((-1, 1))  # torch.Size([996, 1])

# 定义数据加载函数
def load_array(data_arrays, batch_size, is_train=True):
    dataset = TensorDataset(*data_arrays)
    return DataLoader(dataset, batch_size, shuffle=is_train)

# 加载数据
batch_size, n_train = 16, 600
train_iter = load_array((features[:n_train], labels[:n_train]), batch_size)  # DataLoader object

# 定义模型（单层网络）
def get_net():
    net = nn.Sequential(nn.Linear(4, 10),
                        nn.ReLU(),
                        nn.Linear(10, 1))
    net.apply(init_weights)
    return net

# 初始化网络权重的函数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

# 定义损失函数
loss = nn.MSELoss(reduction='none')

# 定义训练函数
def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:  # X: torch.Size([batch_size, 4]), y: torch.Size([batch_size, 1])
            trainer.zero_grad()
            l = loss(net(X), y)  # l: torch.Size([batch_size, 1])
            l.sum().backward()
            trainer.step()
        print(f'epoch {epoch + 1}, loss: {sum(loss(net(X), y)).item() / len(y):f}')

# 训练模型
net = get_net()
train(net, train_iter, loss, 5, 0.01)

# 预测
onestep_preds = net(features) # torch.Size([996, 1])
plt.plot(time, x, label='data')
plt.plot(time[tau:], onestep_preds.detach().numpy(), label='1-step preds')
plt.xlim([1, 1000])
plt.legend()
plt.show()

# k步预测
max_steps = 64
features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))  # torch.Size([933, 68])
# 列i（i<tau）是来⾃x的观测，其时间步从（i）到（i+T-tau-max_steps+1）
for i in range(tau):
    features[:, i] = x[i: i + T - tau - max_steps + 1]
# 列i（i>=tau）是来⾃（i-tau+1）步的预测，其时间步从（i）到（i+T-tau-max_steps+1）
for i in range(tau, tau + max_steps):
    features[:, i] = net(features[:, i - tau:i]).reshape(-1)

steps = (1, 4, 16, 64)
for i in steps:
    plt.plot(time[tau + i - 1: T - max_steps + i], features[:, tau + i - 1].detach().numpy(), label=f'{i}-step preds')
plt.xlim([5, 1000])
plt.legend()
plt.show()