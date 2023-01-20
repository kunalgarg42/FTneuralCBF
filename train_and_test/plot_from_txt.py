import matplotlib.pyplot as plt
import numpy as np
import os
import sys

str = 'log_gamma_test_cbf_u'
file_name = str + '.txt'
plot_name = './plots/' + str + '.png'
f = open(file_name,'r')

traj_len = []
acc_fail = []
acc_no_fail = []

for row in f:
    row = row.split(',')
    if not row[0].isdigit():
        continue
    traj_len.append(int(row[0]))
    acc_fail.append(float(row[1]))
    acc_no_fail.append(float(row[2]))

fig = plt.figure()
ax = fig.subplots(1, 1)

ax.plot(traj_len, acc_fail, color = 'g', label = 'Failture prediction accuracy')
ax.plot(traj_len, acc_no_fail, color = 'r', label = 'No Failture prediction accuracy')

  
plt.xlabel('Length of trajectory with failed actuator', fontsize = 12)
plt.ylabel('Accuracy', fontsize = 12)
  
plt.title('Failure Test Accuracy', fontsize = 20)
plt.legend()
ax.set_xlim(traj_len[0], traj_len[-1])
ax.set_ylim(0.3, 1)
plt.savefig(plot_name)

print("saved file:", str + '.png')