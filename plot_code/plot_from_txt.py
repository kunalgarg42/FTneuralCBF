import matplotlib.pyplot as plt
import numpy as np


# plot_name = './plots/gamma_test_ALL_NN.png'

colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']

# fig = plt.figure(figsize=(10, 8))
# ax1 = fig.subplots(2, 1)

# for i in range(3):
#     if i == 0:
#         str = 'CF_gamma_single_log_only_res_CBF_u'
#     elif i == 1:
#         str = './log_files/log_gamma_test_cbf_u'
#     else:
#         # str = 'CF_gamma_single_log_no_res_LSTM old'
#         str = './log_files/log_gamma_train_CBF_comb_LSTM old'    

#     file_name = str + '.txt'
#     f = open(file_name,'r')

#     traj_len = []
#     acc_fail = []
#     acc_no_fail = []

#     for row in f:
#         row = row.split(',')

#         # a = np.array(literal_eval(row))
        
#         # print(a)

#         # a = np.array(row).astype(np.float)
        
#         # print(a)

#         if not row[0].isdigit():
#             continue

#         traj_len.append(int(row[0]))

#         a = row[1]
#         a = a.replace('[', '').replace(']', '')
                
#         acc_fail.append(float(a))

#         a = row[2]

#         a = a.replace('[', '').replace(']', '')

#         acc_no_fail.append(float(a))


#     markers_on = np.arange(0, traj_len[-1], 10)
#     ax = ax1[0]
#     if i== 0:
#         ax.plot(traj_len, acc_fail, color = colors[i], linestyle='-', label = r"Failure: $(\~ x)$", marker="o", markevery=markers_on,markersize=10)
#         ax.plot(traj_len, acc_no_fail, color = colors[i], linestyle='-', label = r'No Failure: $(\~ x)$', marker="^", markevery=markers_on,markersize=10)
#     elif i==1:
#         ax.plot(traj_len, acc_fail, color = colors[i], linestyle='-', label = r'Failure: $(\~ x, x, u)$', marker="o", markevery=markers_on,markersize=10)
#         ax.plot(traj_len, acc_no_fail, color = colors[i], linestyle='-', label = r'No Failure: $(\~ x, x, u)$', marker="^", markevery=markers_on,markersize=10)
#     else:
#         ax.plot(traj_len, acc_fail, color = colors[i], linestyle='-', label = r'Failure: $(x, u)$', marker="o", markevery=markers_on,markersize=10)
#         ax.plot(traj_len, acc_no_fail, color = colors[i], linestyle='-', label = r'No Failure: $(x, u)$', marker="^", markevery=markers_on,markersize=10)

#     if i==0:
#         str = 'CF_gamma_single_log_only_res'
#     elif i == 1:
#         str = './log_files/log_gamma_test'
#     else:
#         str = './log_files/log_gamma_train_LQR_comb_LSTM old'

#     file_name = str + '.txt'
#     # plot_name = './plots/' + str + 'both.png'
#     f1 = open(file_name,'r')

#     # except: 
#     #     additional_file = 0
#     #     print("no additional file")

#     # if additional_file == 1:
#     traj_len = []
#     acc_fail = []
#     acc_no_fail = []

#     for row in f1:
#         row = row.split(',')
#         if not row[0].isdigit():
#             continue
#         traj_len.append(int(row[0]))
#         # acc_fail.append(float(row[1]))
#         # acc_no_fail.append(float(row[2]))

#         a = row[1]
#         a = a.replace('[', '').replace(']', '')
        
#         acc_fail.append(float(a))

#         a = row[2]
#         a = a.replace('[', '').replace(']', '')
        
#         acc_no_fail.append(float(a))
#     ax2 = ax1[1]

#     if i== 0:
#         ax2.plot(traj_len, acc_fail, color = colors[i], linestyle='--', label = r"Failure: $(\~ x)$", marker="o", markevery=markers_on,markersize=10)
#         ax2.plot(traj_len, acc_no_fail, color = colors[i], linestyle='--', label = r'No Failure: $(\~ x)$', marker="^", markevery=markers_on,markersize=10)
#     elif i==1:
#         ax2.plot(traj_len, acc_fail, color = colors[i], linestyle='--', label = r'Failure: $(\~ x, x, u)$', marker="o", markevery=markers_on,markersize=10)
#         ax2.plot(traj_len, acc_no_fail, color = colors[i], linestyle='--', label = r'No Failure: $(\~ x, x, u)$', marker="^", markevery=markers_on,markersize=10)
#     else:
#         ax2.plot(traj_len, acc_fail, color = colors[i], linestyle='--', label = r'Failure: $(x, u)$', marker="o", markevery=markers_on,markersize=10)
#         ax2.plot(traj_len, acc_no_fail, color = colors[i], linestyle='--', label = r'No Failure: $(x, u)$', marker="^", markevery=markers_on,markersize=10)

# plt.xlabel('Length of trajectory with failed actuator', fontsize = 20)
# ax.set_ylabel('Accuracy (CBF u)', fontsize = 20)
# ax2.set_ylabel('Accuracy (LQR u)', fontsize = 20)

  
# # plt.title('Failure Test Accuracy', fontsize = 20)
# ax.legend(fontsize=20, ncol=1, loc='upper left', bbox_to_anchor=(1.05, 1.0))
# ax2.legend(fontsize=20, ncol=1, loc='upper left', bbox_to_anchor=(1.05, 1.0))
# ax.set_xlim(traj_len[0], traj_len[-1])
# ax2.set_xlim(traj_len[0], traj_len[-1])
# ax.set_ylim(0.5, 1.1)
# ax2.set_ylim(0.5, 1.1)

# # # log y scale
# # ax.set_yscale('log')
# # ax2.set_yscale('log')

# plt.tight_layout()
# ax.tick_params(axis = "x", labelsize = 15)
# ax2.tick_params(axis = "x", labelsize = 15)
# ax.tick_params(axis = "y", labelsize = 15)
# ax2.tick_params(axis = "y", labelsize = 15)

# plt.savefig(plot_name)

# print("saved file:", plot_name)

plot_name = './plots/gamma_test_ALL_NN_new.png'
fig = plt.figure(figsize=(10, 15))
ax1 = fig.subplots(3, 1)

for i in range(3):
    if i == 0:
        str = 'CF_gamma_single_log_only_res_CBF_u'
    elif i == 1:
        str = './log_files/log_gamma_test_cbf_u'
    else:
        # str = 'CF_gamma_single_log_no_res_LSTM old'
        str = './log_files/log_gamma_train_CBF_comb_LSTM old'    

    file_name = str + '.txt'
    f = open(file_name,'r')

    traj_len = []
    acc_fail = []
    acc_no_fail = []

    for row in f:
        row = row.split(',')

        # a = np.array(literal_eval(row))
        
        # print(a)

        # a = np.array(row).astype(np.float)
        
        # print(a)

        if not row[0].isdigit():
            continue

        traj_len.append(int(row[0]))

        a = row[1]
        a = a.replace('[', '').replace(']', '')
                
        acc_fail.append(float(a))

        a = row[2]

        a = a.replace('[', '').replace(']', '')

        acc_no_fail.append(float(a))


    markers_on = np.arange(0, traj_len[-1], 10)
    ax = ax1[i]
    if i== 0:
        ax.plot(traj_len, acc_fail, color = colors[i], linestyle='-', label = r"Failure: $(\~ x)$", marker="o", markevery=markers_on,markersize=10)
        ax.plot(traj_len, acc_no_fail, color = colors[i], linestyle='-', label = r'No Failure: $(\~ x)$', marker="^", markevery=markers_on,markersize=10)
    elif i==1:
        ax.plot(traj_len, acc_fail, color = colors[i], linestyle='-', label = r'Failure: $(\~ x, x, u)$', marker="o", markevery=markers_on,markersize=10)
        ax.plot(traj_len, acc_no_fail, color = colors[i], linestyle='-', label = r'No Failure: $(\~ x, x, u)$', marker="^", markevery=markers_on,markersize=10)
    else:
        ax.plot(traj_len, acc_fail, color = colors[i], linestyle='-', label = r'Failure: $(x, u)$', marker="o", markevery=markers_on,markersize=10)
        ax.plot(traj_len, acc_no_fail, color = colors[i], linestyle='-', label = r'No Failure: $(x, u)$', marker="^", markevery=markers_on,markersize=10)

    if i==0:
        str = 'CF_gamma_single_log_only_res'
    elif i == 1:
        str = './log_files/log_gamma_test'
    else:
        str = './log_files/log_gamma_train_LQR_comb_LSTM old'

    file_name = str + '.txt'
    # plot_name = './plots/' + str + 'both.png'
    f1 = open(file_name,'r')

    # except: 
    #     additional_file = 0
    #     print("no additional file")

    # if additional_file == 1:
    traj_len = []
    acc_fail = []
    acc_no_fail = []

    for row in f1:
        row = row.split(',')
        if not row[0].isdigit():
            continue
        traj_len.append(int(row[0]))
        # acc_fail.append(float(row[1]))
        # acc_no_fail.append(float(row[2]))

        a = row[1]
        a = a.replace('[', '').replace(']', '')
        
        acc_fail.append(float(a))

        a = row[2]
        a = a.replace('[', '').replace(']', '')
        
        acc_no_fail.append(float(a))
    
    ax2 = ax1[i]

    if i== 0:
        ax2.plot(traj_len, acc_fail, color = colors[i], linestyle='--', label = r"Failure: $(\~ x)$", marker="o", markevery=markers_on,markersize=10)
        ax2.plot(traj_len, acc_no_fail, color = colors[i], linestyle='--', label = r'No Failure: $(\~ x)$', marker="^", markevery=markers_on,markersize=10)
    elif i==1:
        ax2.plot(traj_len, acc_fail, color = colors[i], linestyle='--', label = r'Failure: $(\~ x, x, u)$', marker="o", markevery=markers_on,markersize=10)
        ax2.plot(traj_len, acc_no_fail, color = colors[i], linestyle='--', label = r'No Failure: $(\~ x, x, u)$', marker="^", markevery=markers_on,markersize=10)
    else:
        ax2.plot(traj_len, acc_fail, color = colors[i], linestyle='--', label = r'Failure: $(x, u)$', marker="o", markevery=markers_on,markersize=10)
        ax2.plot(traj_len, acc_no_fail, color = colors[i], linestyle='--', label = r'No Failure: $(x, u)$', marker="^", markevery=markers_on,markersize=10)

plt.xlabel('Length of trajectory with failed actuator', fontsize = 20)
ax.set_ylabel('Accuracy (CBF u)', fontsize = 20)
ax2.set_ylabel('Accuracy (LQR u)', fontsize = 20)

  
# plt.title('Failure Test Accuracy', fontsize = 20)
ax.legend(fontsize=20, ncol=1, loc='upper left', bbox_to_anchor=(1.05, 1.0))
ax2.legend(fontsize=20, ncol=1, loc='upper left', bbox_to_anchor=(1.05, 1.0))
ax.set_xlim(traj_len[0], traj_len[-1])
ax2.set_xlim(traj_len[0], traj_len[-1])
ax.set_ylim(0.5, 1.1)
ax2.set_ylim(0.5, 1.1)

# # log y scale
# ax.set_yscale('log')
# ax2.set_yscale('log')

plt.tight_layout()
ax.tick_params(axis = "x", labelsize = 15)
ax2.tick_params(axis = "x", labelsize = 15)
ax.tick_params(axis = "y", labelsize = 15)
ax2.tick_params(axis = "y", labelsize = 15)

plt.savefig(plot_name)

print("saved file:", plot_name)