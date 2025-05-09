# import torch
# from statsmodels.graphics.tukeyplot import results
#
# x = torch.arange(12)
# print(x)
#
# print(x.shape)
# print(x.numel())
#
# x = x.reshape(3, 4)
# print(x)
#
#
# # 创建一个（2，3，4）的张量
# print(torch.zeros(2,3,4))
#
# # 创建一个全部1的张量
# print(torch.ones(2,3,4))
#
# # 随机生成，创建一个0-1高斯分布张量
# print(torch.randn(3,4))
#
# #
# print(torch.tensor([[2,1,4,3],
#                     [1,2,3,4],
#                     [4,3,2,1]]))
#
# x = torch.tensor([1.0,2,4,8])
# y = torch.tensor([2,2,2,2])
# print(x+y)


import math
import torch

# Sigmoid这是一个激活函数（神经元）
def sigmoid(x):
    return 1/(1+math.exp(-x))

x = 0.5
result = sigmoid(x)
print(result)

if __name__ == '__main__':
    x = torch.randn(1)
    print(x)


x = torch.arange(12)
print(x)

x.shape
print(x.size())
















