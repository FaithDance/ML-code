import sys
import time
import numpy
import torch
import torch.nn as nn
import torch.optim as optim

import my_app
import matplotlib.pyplot as plt

torch.manual_seed(23342)
torch.set_printoptions(profile = "full")
# 检查是否有可用的 GPU
if torch.cuda.is_available():
  device = torch.device("cuda")  # 使用第一个可用的 GPU
  print("Using GPU:", torch.cuda.get_device_name(0))
else:
  device = torch.device("cpu")
  print("Using CPU")

saved_stderr = sys.stderr
sys.stderr = open("log.txt", "w")

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



if __name__ == '__main__':
  # 定义网络参数
  input_size = 128
  output_size = 15
  hidden_size = 5
  num_hidden_layers = 3  # k个隐藏层

  # 创建模型实例
  model = myNN(input_size, hidden_size, output_size, num_hidden_layers)
  model = model.double()
  print(model)
  # 将模型移动到 GPU（如果可用）
  model = model.to(device)

  # 定义损失函数和优化器
  criterion = nn.MSELoss()
  optimizer = optim.Adam(model.parameters(), lr = 0.0001)

  # ------------------------------------------------------------------------------------------------ #

  # 导入训练数据
  train_inputs = torch.tensor(numpy.loadtxt('_train_inputs.txt'), dtype = torch.double)
  train_targets = torch.tensor(numpy.loadtxt('_train_targets.txt'), dtype = torch.double)
  num_train = train_inputs.shape[0]
  print(f'train data shape: {train_inputs.shape}, {train_targets.shape}')
  # 将训练数据移动到 GPU
  train_inputs = train_inputs.to(device)
  train_targets = train_targets.to(device)


  print('start training...')
  # 训练网络
  num_epochs = 7  # 训练轮数
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
      if (step % 1000 == 0):
        print(f'  percent: {int(step / (num_train / batch_size) * 100)}%')
        print(f'    > Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.7f}')
        loss_plt.append(loss.item())
    # 打印损失
    if (epoch + 1) % 1 == 0:
      print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.7f}')
      loss_plt.append(loss.item())

  torch.save(model, 'model.pth')
  torch.save(model.state_dict(), 'model_state_dict.pth')

  # 导入测试数据
  test_inputs = torch.tensor(numpy.loadtxt('_test_inputs.txt'), dtype = torch.double)
  test_targets = torch.tensor(numpy.loadtxt('_test_targets.txt'), dtype = torch.double)
  num_test = test_inputs.shape[0]
  print(f'test data shape: {test_inputs.shape}, {test_targets.shape}')
  # 将测试数据移动到 GPU
  test_inputs = test_inputs.to(device)
  test_targets = test_targets.to(device)

  # 将模型设置为推理模式
  model.eval()

  time_start = time.perf_counter()

  # 前向传播
  with torch.no_grad():
    test_outputs = model(test_inputs)
    test_loss = criterion(test_outputs, test_targets)

  time_end = time.perf_counter()
  print('Running time: %s Seconds' % (time_end - time_start))

  # 打印测试损失
  print(f'Test Loss: {test_loss.item():.4f}')

  my_app.draw(torch.arange(0, len(loss_plt)), loss_plt, 'loss')
  plt.show()