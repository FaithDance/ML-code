import matplotlib.pyplot as plt

def draw(x, y, title, if_ylim_01 = 0):
  plt.figure(figsize = (8, 5))
  plt.plot(x, y)
  if if_ylim_01:
    plt.ylim(0, 1)
  plt.title(title)

def draw_nk(x, n, k, title):
  plt.figure(figsize = (8, 5))
  plt.plot(x, n, label = 'n')
  plt.plot(x, k, label = 'k')
  plt.title(title)
  plt.legend()
