# import torch
#
# name = "Tim"
# location = "China"
#
# formatted_string = f"My-minist name is {name}, from {location}"
#
# # 转换为ASCII
# ascii_values = [ord(char) for char in formatted_string]
# ascii_matrix = torch.tensor(ascii_values)  # 添加一维以形成矩阵
#
#
# # 输出矩阵
# print(ascii_matrix)
#
#
# # 定义变量
# name = 'Tim'
# location = 'China'
#
# # %s f .format
# print(f'My name is {name}, from {location}')
# print('My name is %s, from %s' % (name, location))
# print('My name is {}, from {}'.format(name,location))
#
#
#
# # 定义 sigmoid 激活函数
# def sigmoid(x):
#     return 1 / (1 + torch.exp(-x))
#
#
# name = "Tim"
# location = "China"
#
# # 格式化字符串
# formatted_string = f"My-minist name is {name}, from {location}"
#
# # 转换为 ASCII 码
# ascii_values = [ord(char) for char in formatted_string]
#
# # 创建张量
# ascii_tensor = torch.tensor(ascii_values, dtype=torch.float32)
#
# # 应用 sigmoid 函数
# output_tensor = sigmoid(ascii_tensor)
#
# # 输出结果
# print(output_tensor)
#
#
#
# # 定义 sigmoid 激活函数
# def sigmoid(x):
#     return 1 / (1 + torch.exp(-x))
#
# name = "Tim"
# location = "China"
#
# # 格式化字符串
# formatted_string = f"My-minist name is {name}, from {location}"
#
# # 转换为 ASCII 码
# ascii_values = [ord(char) for char in formatted_string]
#
# # 创建张量
# ascii_tensor = torch.tensor(ascii_values, dtype=torch.float32)
#
# # 标准化输入，例如减去 64（因为ASCII值范围从0到127）
# normalized_tensor = ascii_tensor - 64
#
# # 应用 sigmoid 函数
# output_tensor = sigmoid(normalized_tensor)
#
# # 输出结果
# print(output_tensor)

# import tensorflow as tf
# from tensorflow.keras import lazy,model
# from tensorflow.keras.datasets import mnist
# from tensorflow.keras.utils import to_categorical
# import numpy as np
# import matplotlib.pyplot as plt
#
# (train_images,train_labels),(test_images, test_labels)=mnist.load_data()


class MyTorch:
    def __init__(self, nums):
        self.nums = nums

    @staticmethod
    def arrages(nums):
        r"""
        any(input) -> Tensor

        Tests if any element in :attr:`input` evaluates to `True`.
        .. note:: This function matches the behaviour of NumPy in returning
            output of dtype `bool` for all supported dtypes except `uint8`.
            For `uint8` the dtype of output is `uint8` itself.

        this is a function,you can input an int number like: a = 1
        """
        # result_list = []
        # for num in range(nums):
        #     result_list.append(num)
        # print(result_list)
        # return result_list

        return list(range(nums))

MyTorch.arrages(12)


