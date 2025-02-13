import sys
import torch
import torch.nn as nn
import torch.optim as optim

import my_app
import matplotlib.pyplot as plt

torch.manual_seed(23342)
# 检查是否有可用的 GPU
if torch.cuda.is_available():
  device = torch.device("cuda")  # 使用第一个可用的 GPU
  print("Using GPU:", torch.cuda.get_device_name(0))
else:
  device = torch.device("cpu")
  print("Using CPU")


# 定义一个包含多个隐藏层的全连接神经网络
class myNN(nn.Module):
  def __init__(self, input_size, hidden_size, output_size, num_hidden_layers):
    super(myNN, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.num_hidden_layers = num_hidden_layers
    # 创建输入层到第一个隐藏层
    self.layers = nn.ModuleList([nn.Linear(input_size, hidden_size)])
    # 创建中间的隐藏层
    for _ in range(num_hidden_layers - 1):
      self.layers.append(nn.Linear(hidden_size, hidden_size))
    # 创建最后一个隐藏层到输出层
    self.layers.append(nn.Linear(hidden_size, output_size))
    # 激活函数
    self.relu = nn.ReLU()

  def forward(self, x):
    for i, layer in enumerate(self.layers[:-1]):
      x = self.relu(layer(x))
    x = self.layers[-1](x)  # 最后一层不使用激活函数
    return x


# 随便构造一个函数，让网络来拟合
def func(x):
  A = torch.sum(x)
  B = torch.norm(x)
  C = torch.sum(x ** 3)
  D = torch.log(x ** 2 + 1).sum()
  E = torch.arccos(torch.dot(x, torch.flip(x, [0])) / (torch.norm(x) ** 2 + 1e-7))
  return torch.tensor([A, B, C, D, E])



# 定义网络参数
input_size = 10
hidden_size = 256
output_size = 5
num_hidden_layers = 5  # k个隐藏层

# 创建模型实例
model = myNN(input_size, hidden_size, output_size, num_hidden_layers)
print(model)
# 将模型移动到 GPU（如果可用）
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.0001)

# ------------------------------------------------------------------------------------------------ #

# 生成一些随机数据用于训练
num_train = 250000
train_inputs = torch.randn(num_train, input_size)
train_targets = torch.zeros(num_train, output_size)
for i in range(num_train):
  train_targets[i] = func(train_inputs[i])
# 将数据移动到 GPU
train_inputs = train_inputs.to(device)
train_targets = train_targets.to(device)



# 训练网络
num_epochs = 100  # 训练轮数
batch_size = 42
loss_plt = []
for epoch in range(num_epochs):
  for step in range(int(num_train / batch_size)):
    tl = batch_size * step
    tr = batch_size * (step + 1)
    # 前向传播
    outputs = model(train_inputs[tl : tr])
    loss = criterion(outputs, train_targets[tl : tr])
    # 反向传播和优化
    optimizer.zero_grad()  # 清零梯度
    loss.backward()  # 反向传播
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
    optimizer.step()  # 更新参数
  # 打印损失
  if (epoch + 1) % 1 == 0:
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.7f}')
    loss_plt.append(loss.item())


# 生成一些随机数据用于测试
# 假设我们有20个样本，每个样本有10个输入参数和5个输出参数
num_test = 20
test_inputs = torch.randn(num_test, input_size)
test_targets = torch.zeros(num_test, output_size)
for i in range(num_test):
  test_targets[i] = func(test_inputs[i])
# 将测试数据移动到 GPU
test_inputs = test_inputs.to(device)
test_targets = test_targets.to(device)

# 将模型设置为推理模式
model.eval()

# 前向传播
with torch.no_grad():
  test_outputs = model(test_inputs)
  test_loss = criterion(test_outputs, test_targets)

# 打印测试损失
print(test_outputs)
print(f'Test Loss: {test_loss.item():.4f}')
print(test_inputs[-1])
print(test_targets[-1])
print(model(test_inputs[-1]))

my_app.draw(torch.arange(0, num_epochs), loss_plt, 'loss')
plt.show()