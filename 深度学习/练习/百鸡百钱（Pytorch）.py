import torch
import torch.nn as nn
import torch.optim as optim
import random


# 定义多层感知机模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(1, 64)  # 输入层到隐藏层  (公鸡)
        self.layer2 = nn.Linear(64, 64)  # 隐藏层到隐藏层
        self.layer3 = nn.Linear(64, 2)  # 隐藏层到输出层（母鸡、小鸡的数量）

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)  # 输出层不使用激活函数，因为我们需要逼近连续值
        return x


# 生成所有满足条件的（公鸡，母鸡，小鸡）组合
def generate_combinations():
    combinations = []
    for cock in range(21):  # 公鸡最多20只（5钱一只，最多100钱）
        for hen in range(34):  # 母鸡最多33只（3钱一只，最多99钱，留一只小鸡的可能性）
            chick = 100 - cock - hen
            if chick % 3 == 0 and 5 * cock + 3 * hen + chick / 3 == 100:
                combinations.append((cock, hen, chick))
    return combinations


# 从满足条件的组合中随机选择一部分作为训练数据
def select_random_samples(combinations, num_samples):
    random_samples = random.sample(combinations, num_samples)
    inputs = [sample[0] for sample in random_samples]  # 公鸡数量作为输入
    outputs = [(sample[1], sample[2]) for sample in random_samples]  # 母鸡和小鸡数量作为输出
    return torch.tensor(inputs, dtype=torch.float32), torch.tensor(outputs, dtype=torch.float32)


# 训练模型
def train_model(model, data, labels, epochs=10000, lr=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')


# 简单的任务，极致的享受（直接过拟合，无需泛化）
model = MLP()
combinations = generate_combinations()
num_samples = len(combinations)  # 样本数量
data, labels = select_random_samples(combinations, num_samples)
print(f"data:{data}")
print(f"labels:{labels}")
train_model(model, data[:, None], labels, epochs=1500, lr=0.001)

# 批量测试数据
test_data = torch.tensor([[0], [4], [8], [12]], dtype=torch.float32)
for data in test_data:
    # 测试无需梯度
    with torch.no_grad():
        # 测试模型
        # test_input = torch.tensor([4], dtype=torch.float32)  # 输入
        prediction = model(data)
        rounded_prediction = torch.round(prediction).int()  # 对输出进行四舍五入并取整
        公鸡 = int(data.item())
        母鸡 = rounded_prediction.numpy()[0]
        小鸡 = rounded_prediction.numpy()[1]
        print("公鸡的数量:", 公鸡)
        print("预测母鸡的数量:", 母鸡)
        print("预测小鸡的数量:", 小鸡)
        # 检测
        print(f"{公鸡}×5+{母鸡}×3+{小鸡}÷3={int(公鸡 * 5 + 母鸡 * 3 + 小鸡 / 3)}")
