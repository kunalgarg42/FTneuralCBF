import matplotlib.pyplot as plt
import numpy as np
import re
from ast import literal_eval

str = 'CF_gamma_single_train_log'
file_name = str + '.txt'
plot_name = './plots/' + str + '.png'
f = open(file_name,'r')

step = []
loss = []
acc = []
acc_fail = []

for row in f:
    row = row.split(', ')
    if len(row) < 5 or not row[1].isdigit():
        continue
    step.append(int(row[1]))
    loss.append(float(row[3]))
    acc.append(float(row[5]))
    a = row[7]
    a = re.sub(r"([^[])\s+([^]])", r"\1, \2", a)
    a = np.array(literal_eval(a))
    acc_fail.append(a[1])

fig = plt.figure()
ax = fig.subplots(1, 1)

ax.plot(step, loss, color = 'g', label = 'Training loss')
ax.set_ylim(0, np.array(loss).max()+0.2)
ax.set_ylabel('Loss', fontsize = 12, color = 'g')
ax.legend()

h_ax = ax.twinx()
h_ax.plot(step, acc, color = 'r', label = 'Overall Accuracy')
h_ax.plot(step, acc_fail, color = 'r', label = 'Fail Prediction Accuracy')

plt.xlabel('Training epochs', fontsize = 12)
h_ax.set_ylabel('Accuracy', fontsize = 12, color = 'r')
  
plt.title('Training log', fontsize = 20)


h_ax.legend(loc="upper center")

h_ax.set_xlim(step[0], 100)
h_ax.set_ylim(0, 1.2)

plt.savefig(plot_name)

print("saved file:", plot_name)