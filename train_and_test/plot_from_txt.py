import matplotlib.pyplot as plt

str = 'CF_gamma_single_log_only_res_CBF_u'
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

fig = plt.figure(figsize=(10, 6))
ax = fig.subplots(1, 1)

ax.plot(traj_len, acc_fail, color = 'g', label = 'Failure (CBF)')
ax.plot(traj_len, acc_no_fail, color = 'r', label = 'No Failure (CBF)')


try:
    str = 'CF_gamma_single_log_only_res'
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

    ax.plot(traj_len, acc_fail, color = 'g', linestyle='--', label = 'Failure (LQR)')
    ax.plot(traj_len, acc_no_fail, color = 'r', linestyle='--', label = 'No Failure (LQR)')

plt.xlabel('Length of trajectory with failed actuator', fontsize = 20)
plt.ylabel('Accuracy', fontsize = 20)
  
plt.title('Failure Test Accuracy', fontsize = 20)
plt.legend(fontsize=15, ncol =2)
ax.set_xlim(traj_len[0], traj_len[-1])
ax.set_ylim(0.6, 1)
ax.tick_params(axis = "x", labelsize = 15)
ax.tick_params(axis = "y", labelsize = 15)

plt.savefig(plot_name)

print("saved file:", plot_name)