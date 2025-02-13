import sys
import time
import nlopt
import autograd
import autograd.numpy as np

import my_app
import matplotlib.pyplot as plt

np.random.seed(int(time.time()))

N = 128
NN = (N + 1) * N
# alpha, N vector
# beta, N * N matrix


# 导入三个标准光谱
mat_std_1 = np.loadtxt('data\\G_plant_ROI_400-1000.txt')
mat_std_2 = np.loadtxt('data\\W_plant_ROI_400-1000.txt')
mat_std_3 = np.loadtxt('data\\YG_plant_ROI_400-1000.txt')
lam = mat_std_1[:, 0]
I_std_1 = mat_std_1[:, 1]
I_std_2 = mat_std_2[:, 1]
I_std_3 = mat_std_3[:, 1]


# Ia, Ib 都是向量，并非 1 * N 的矩阵
def GCD(Ia, Ib, alpha, beta, kk, bb):
  a = Ia / Ia.sum()
  b = Ib / Ib.sum()
  SID = np.dot(a, np.log10(a / b)) + np.dot(b, np.log10(b / a))
  SIDp = kk * SID + bb
  SAM = np.arccos(np.dot(Ia, Ib) / np.sqrt(np.dot(Ia, Ia) * np.dot(Ib, Ib)))
  SWD = np.sqrt(np.dot(alpha, (Ia - Ib) ** 2) + np.dot(np.dot(Ia, beta), Ib))
  GCD = np.sqrt(SWD ** 2 + (SIDp * np.sin(SAM)) ** 2)
  return GCD


# 优化的目标函数
def func_target(para):
  alpha = para[0 : N]
  beta = para[N : NN]
  beta = beta.reshape(N, N)
  kk, bb = para[NN], para[NN + 1]
  # print(f'sizeof alpha = {alpha.shape}, sizeof beta = {beta.shape}')
  GCD12 = GCD(I_std_1, I_std_2, alpha, beta, kk, bb)
  GCD13 = GCD(I_std_1, I_std_3, alpha, beta, kk, bb)
  GCD23 = GCD(I_std_2, I_std_3, alpha, beta, kk, bb)
  mx = max(max(GCD12, GCD13), GCD23)
  mn = min(min(GCD12, GCD13), GCD23)
  return mx - mn

counter = 0
grad_target = autograd.grad(func_target)

def func_nlopt(para, grad):
  global counter
  grad[:] = grad_target(para)
  res = func_target(para)
  # print(f'Step = {counter}, res = {res}')
  print(f'Step = {counter}, res = {res}', file = sys.stderr)
  counter += 1
  return res


if __name__ == '__main__':
  # 将标准错误流重定向到 log.txt
  saved_stderr = sys.stderr
  log_file = open("log.txt", "w")
  sys.stderr = log_file
  time_start = time.perf_counter()
  # ---------------------------

  if (True):
    # print(I_std_1, '\n', I_std_2, '\n', I_std_3)
    alpha_0 = np.full(N, 0.5)
    beta_0 = np.full(N * N, 0).reshape(N, N)
    print(GCD(I_std_1, I_std_2, alpha_0, beta_0, 50, 0))
    print(GCD(I_std_1, I_std_3, alpha_0, beta_0, 50, 0))
    print(GCD(I_std_2, I_std_3, alpha_0, beta_0, 50, 0))
    # exit()

  print('start opt...')
  lower_bd = np.concatenate((np.full(N, 0.3), np.full(N * N, 0.0), np.array([40, -10])))
  upper_bd = np.concatenate((np.full(N, 0.7), np.full(N * N, 100), np.array([60, 10])))
  opt = nlopt.opt(nlopt.LD_MMA, NN + 2)
  opt.set_lower_bounds(lower_bd)
  opt.set_upper_bounds(upper_bd)
  opt.set_xtol_rel(1e-5)
  opt.set_maxeval(233)
  opt.set_min_objective(func_nlopt)

  init_para = np.concatenate((np.full(N, 0.5), np.full(N * N, 0), np.array([50, 0])))
  best_para = init_para
  for rd in range(0, 42):
    print(f'round: {rd}')
    print(f'round: {rd}', file = sys.stderr)
    counter = 0
    b_para = opt.optimize(init_para)
    if (func_target(b_para) < func_target(best_para)):
      best_para = b_para
    next_kb = np.random.random(2) * (-np.array([40, -10]) + np.array([60, 10])) + np.array([40, -10])
    init_para = np.concatenate((np.full(N, 0.5), np.full(N * N, 0), next_kb))

  # print(f'beat = {best_para}')
  # print(f'best = {best_para}', file = sys.stderr)
  alpha_o = best_para[0 : N]
  beta_o = best_para[N : NN]
  beta_o = beta_o.reshape(N, N)
  kk_o, bb_o = best_para[NN], best_para[NN + 1]
  with np.printoptions(threshold = np.inf):
    print(alpha_o, file = sys.stderr)
    print(beta_o, file = sys.stderr)
    print(kk_o, file = sys.stderr)
    print(bb_o, file = sys.stderr)
    print(f'max in beta = {beta_o.max()}')

  print(GCD(I_std_1, I_std_2, alpha_o, beta_o, kk_o, bb_o))
  print(GCD(I_std_1, I_std_3, alpha_o, beta_o, kk_o, bb_o))
  print(GCD(I_std_2, I_std_3, alpha_o, beta_o, kk_o, bb_o))

  time_end = time.perf_counter()
  print('Running time: %s Seconds' % (time_end - time_start))