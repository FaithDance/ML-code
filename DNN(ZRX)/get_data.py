import sys
import time
import numpy
import pandas
import torch


N = 128
# 导入三个标准光谱
mat_std_1 = torch.tensor(numpy.loadtxt('data\\G_plant_ROI_400-1000.txt'), dtype = torch.double)
mat_std_2 = torch.tensor(numpy.loadtxt('data\\W_plant_ROI_400-1000.txt'), dtype = torch.double)
mat_std_3 = torch.tensor(numpy.loadtxt('data\\YG_plant_ROI_400-1000.txt'), dtype = torch.double)
I_std_1 = mat_std_1[:, 1]
I_std_2 = mat_std_2[:, 1]
I_std_3 = mat_std_3[:, 1]

alpha = torch.tensor(numpy.loadtxt('_alpha.txt').flatten(), dtype = torch.double)
beta = torch.tensor(numpy.loadtxt('_beta.txt').reshape(N, N), dtype = torch.double)
kk = 48.65681411398964
bb = -2.3746861023905295
print(f'sizeof alpha & beta: {alpha.shape}, {beta.shape}')


# SD, SAM, SID, SWD, GCD
# Ia, Ib, N vector
def SD(Ia, Ib):
  return torch.sqrt(torch.dot(Ia - Ib, Ia - Ib))
def SAM(Ia, Ib):
  return torch.arccos(torch.dot(Ia, Ib) / torch.sqrt(torch.dot(Ia, Ia) * torch.dot(Ib, Ib)))
def SID(Ia, Ib):
  a = torch.clamp(Ia / Ia.sum(), min = 1e-7)
  b = torch.clamp(Ib / Ib.sum(), min = 1e-7)
  return torch.dot(a, torch.log10(a / b)) + torch.dot(b, torch.log10(b / a))
def SIDp(Ia, Ib):
  return kk * SID(Ia, Ib) + bb
def SWD(Ia, Ib):
  return torch.sqrt(torch.dot(alpha, (Ia - Ib) ** 2) + torch.dot(torch.matmul(Ia, beta), Ib))
def GCD(Ia, Ib):
  return torch.sqrt(SWD(Ia, Ib) ** 2 + (SIDp(Ia, Ib) * torch.sin(SAM(Ia, Ib))) ** 2)

# 网络需要拟合的函数
# in 128 -> 15 out
def func(I):
  res_1 = torch.tensor([SD(I, I_std_1), SAM(I, I_std_1), SID(I, I_std_1), SWD(I, I_std_1), GCD(I, I_std_1)])
  res_2 = torch.tensor([SD(I, I_std_2), SAM(I, I_std_2), SID(I, I_std_2), SWD(I, I_std_2), GCD(I, I_std_2)])
  res_3 = torch.tensor([SD(I, I_std_3), SAM(I, I_std_3), SID(I, I_std_3), SWD(I, I_std_3), GCD(I, I_std_3)])
  return torch.cat((res_1, res_2, res_3))


if __name__ == '__main__':
  print('loading data...')
  data_csv = pandas.read_csv('data\\DNN\\All.csv', header = None, usecols = range(2, 130))
  print(data_csv)
  data_csv = torch.tensor(data_csv.values, dtype = torch.double)
  num_data = data_csv.shape[0]
  print(num_data)

  time_start = time.perf_counter()
  select_test = (torch.arange(num_data) % 20 == 0)

  train_inputs = data_csv[~select_test]
  num_train = train_inputs.shape[0]
  train_targets = torch.zeros(num_train, 15, dtype = torch.double)
  for i in range(0, num_train):
    train_targets[i] = func(train_inputs[i])
    if (i % 10000 == 0):
      print(f'train data calculating: {i} ...')
  print(f'train data shape: {train_inputs.shape}, {train_targets.shape}')
  numpy.savetxt('_train_inputs.txt', train_inputs.numpy())
  numpy.savetxt('_train_targets.txt', train_targets.numpy())

  test_inputs = data_csv[select_test]
  num_test = test_inputs.shape[0]
  test_targets = torch.zeros(num_test, 15, dtype = torch.double)
  for i in range(0, num_test):
    test_targets[i] = func(test_inputs[i])
    if (i % 10000 == 0):
      print(f'test data calculating: {i} ...')
  print(f'test data shape: {test_inputs.shape}, {test_targets.shape}')
  numpy.savetxt('_test_inputs.txt', test_inputs.numpy())
  numpy.savetxt('_test_targets.txt', test_targets.numpy())

  time_end = time.perf_counter()
  print('Running time: %s Seconds' % (time_end - time_start))