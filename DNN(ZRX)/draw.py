import sys
import numpy

import my_app
import matplotlib.pyplot as plt


train_targets = numpy.loadtxt('_train_targets.txt')
num_train = train_targets.shape[0]
print(f'train data shape: {train_targets.shape}')
test_targets = numpy.loadtxt('_test_targets.txt')
num_test = test_targets.shape[0]
print(f'test data shape: {test_targets.shape}')

targets = numpy.vstack((train_targets, test_targets))
print(f'data shape: {targets.shape}')
numpy.savetxt('_all_targets.txt', targets)

data_SD = targets[:, 0]
data_SAM = targets[:, 1]
data_SID = targets[:, 2]
data_SWD = targets[:, 3]
data_GCD = targets[:, 4]
print(f'SD shape: {data_SD.shape}')

plt.figure(1)
plt.hist(data_SD, bins = 233, alpha = 0.75, color = 'blue', edgecolor = 'black')
plt.title('SD - Histogram')
plt.xlabel('SD')
plt.ylabel('Number')

plt.figure(2)
plt.hist(data_SAM, bins = 233, alpha = 0.75, color = 'blue', edgecolor = 'black')
plt.title('SAM - Histogram')
plt.xlabel('SAM')
plt.ylabel('Number')

plt.figure(3)
plt.hist(data_SID, bins = 233, alpha = 0.75, color = 'blue', edgecolor = 'black')
plt.title('SID - Histogram')
plt.xlabel('SID')
plt.ylabel('Number')

plt.figure(4)
plt.hist(data_SWD, bins = 233, alpha = 0.75, color = 'blue', edgecolor = 'black')
plt.title('SWD - Histogram')
plt.xlabel('SWD')
plt.ylabel('Number')

plt.figure(5)
plt.hist(data_GCD, bins = 233, alpha = 0.75, color = 'blue', edgecolor = 'black')
plt.title('GCD - Histogram')
plt.xlabel('GCD')
plt.ylabel('Number')

plt.show()