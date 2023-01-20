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

ax.plot(traj_len, acc_fail, color = 'g', label = 'Failture prediction accuracy (CBF)')
ax.plot(traj_len, acc_no_fail, color = 'r', label = 'No Failture prediction accuracy (CBF)')


try:
    str = 'log_gamma_test'
    file_name = str + '.txt'
    plot_name = './plots/' + str + 'both.png'
    f1 = open(file_name,'r')
    additional_file = 1

except: 
    additional_file = 0
    print("no additional file")

if additional_file == 1:
    traj_len = []
    acc_fail = []
    acc_no_fail = []

    for row in f1:
        row = row.split(',')
        if not row[0].isdigit():
            continue
        traj_len.append(int(row[0]))
        acc_fail.append(float(row[1]))
        acc_no_fail.append(float(row[2]))

    ax.plot(traj_len, acc_fail, color = 'g', linestyle='--', label = 'Failture prediction accuracy (LQR)')
    ax.plot(traj_len, acc_no_fail, color = 'r', linestyle='--', label = 'No Failture prediction accuracy (LQR)')

plt.xlabel('Length of trajectory with failed actuator', fontsize = 12)
plt.ylabel('Accuracy', fontsize = 12)
  
plt.title('Failure Test Accuracy', fontsize = 20)
plt.legend()
ax.set_xlim(traj_len[0], traj_len[-1])
ax.set_ylim(0.3, 1)

plt.savefig(plot_name)

print("saved file:", plot_name)