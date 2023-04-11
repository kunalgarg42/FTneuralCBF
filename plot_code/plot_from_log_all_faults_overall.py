import matplotlib.pyplot as plt
import numpy as np
import re
from ast import literal_eval

colors = ['g', 'r', 'b', 'k']
for j in range(4):
    str1 = 'log_gamma_test'
    file_name = str1 + '_nom_u_rotate_FI_' + str(j) + '.txt'
    plot_name = './plots/' + str1 + '_rotate_FI_overall_' + str(j) + '.png'
    f = open(file_name,'r')

    step = []

    acc0 = []
    acc1 = []
    acc2 = []
    acc3 = []
    
    acc0_no_fail = []
    acc1_no_fail = []
    acc2_no_fail = []
    acc3_no_fail = []
    
    for row in f:
        row = row.split(', ')
        if not row[0].isdigit():
            continue
        step.append(int(row[0]))

        a = row[1]
        a = re.sub(r"([^[])\s+([^]])", r"\1, \2", a)
        a = np.array(literal_eval(a))

        acc0.append(a[j])
        acc0_no_fail.append(a[-1])

        # a = row[2]
        # a = re.sub(r"([^[])\s+([^]])", r"\1, \2", a)
        # a = np.array(literal_eval(a))
        # acc1.append(a[j])
        # acc1_no_fail.append(a[-1])

        # a = row[3]
        # a = re.sub(r"([^[])\s+([^]])", r"\1, \2", a)
        # a = np.array(literal_eval(a))
        # acc2.append(a[j])
        # acc2_no_fail.append(a[-1])

        # a = row[4]
        # a = re.sub(r"([^[])\s+([^]])", r"\1, \2", a)
        # a = np.array(literal_eval(a))
        # acc3.append(a[j])
        # acc3_no_fail.append(a[-1])

    fig = plt.figure(figsize=(11.5, 7))
    axes = fig.subplots(1, 1)
    
    ax = axes

    markers_on = np.arange(0, step[-1], 10)

    ax.plot(step, acc0, color = colors[j], label = 'LQR input (Fault in motor {})'.format(j + 1), marker="o", markevery=markers_on,markersize=10)

    # ax.plot(step, acc1, color = 'r', label = 'LQR input (Case 2)', marker="^", markevery=markers_on,markersize=10)

    # ax.plot(step, acc2, color = 'b', label = 'LQR input (Case 3)', marker="v", markevery=markers_on,markersize=10)

    # ax.plot(step, acc3, color = 'k', label = 'LQR input (Case 4)', marker="<", markevery=markers_on,markersize=10)

    ax.set_xlim(step[0], 100)

    ax.set_ylim(0.5, 1)
    ax.set_ylabel('Accuracy', fontsize = 30)
    ax.set_xlabel('Duration of failed actuator', fontsize = 30)

    # ax.set_xlabel('                                  Duration of failed actuator', fontsize = 30)

    # ax.legend(ncol = 2, fontsize = 15)

    # h_ax = axes[1]

    # h_ax.plot(step, acc0_no_fail, color = 'g', label = 'LQR input (Case 1)')

    # h_ax.plot(step, acc1_no_fail, color = 'r', label = 'LQR input (Case 2)')

    # h_ax.plot(step, acc2_no_fail, color = 'b', label = 'LQR input (Case 3)')

    # h_ax.plot(step, acc3_no_fail, color = 'k', label = 'LQR input (Case 4)')

    file_name = str1 + '_cbf_u_rotate_FI_' + str(j) + '.txt'
    f = open(file_name,'r')

    step = []
    loss = []
    acc0 = []
    acc1 = []
    acc2 = []
    acc3 = []
    
    acc0_no_fail = []
    acc1_no_fail = []
    acc2_no_fail = []
    acc3_no_fail = []
    
    for row in f:
        row = row.split(', ')
        if not row[0].isdigit():
            continue
        step.append(int(row[0]))
        a = row[1]
        a = re.sub(r"([^[])\s+([^]])", r"\1, \2", a)
        a = np.array(literal_eval(a))
        acc0.append(a[j])
        acc0_no_fail.append(a[-1])

        # a = row[2]
        # a = re.sub(r"([^[])\s+([^]])", r"\1, \2", a)
        # a = np.array(literal_eval(a))
        # acc1.append(a[j])
        # acc1_no_fail.append(a[-1])

        # a = row[3]
        # a = re.sub(r"([^[])\s+([^]])", r"\1, \2", a)
        # a = np.array(literal_eval(a))
        # acc2.append(a[j])
        # acc2_no_fail.append(a[-1])

        # a = row[4]
        # a = re.sub(r"([^[])\s+([^]])", r"\1, \2", a)
        # a = np.array(literal_eval(a))
        # acc3.append(a[j])
        # acc3_no_fail.append(a[-1])

    ax.plot(step, acc0, color = colors[j], linestyle='--',  label = 'CBF input (Fault in motor {})'.format(j + 1), marker="o", markevery=markers_on,markersize=10)

    # ax.plot(step, acc1, color = 'r', linestyle='--',  label = 'CBF input (Case 2)', marker="^", markevery=markers_on,markersize=10)

    # ax.plot(step, acc2, color = 'b', linestyle='--', label = 'CBF input (Case 3)', marker="v", markevery=markers_on,markersize=10)

    # ax.plot(step, acc3, color = 'k', linestyle='--', label = 'CBF input (Case 4)', marker="<", markevery=markers_on,markersize=10)

    # ax.set_title('Failure prediction for Actuator-' + str(j + 1), fontsize = 20, loc= 'center')

    

    # h_ax.plot(step, acc0_no_fail, color = 'g', linestyle='--', label = 'CBF input (Case 1)')

    # h_ax.plot(step, acc1_no_fail, color = 'r', linestyle='--', label = 'CBF input (Case 2)')

    # h_ax.plot(step, acc2_no_fail, color = 'b', linestyle='--', label = 'CBF input (Case 3)')

    # h_ax.plot(step, acc3_no_fail, color = 'k', linestyle='--', label = 'CBF input (Case 4)')

    # # h_ax.set_xlabel('Duration of failed actuator', fontsize = 20)
    
    # # h_ax.legend(ncol = 2,  loc = 'lower center')

    # h_ax.set_ylim(-0.1, 1)
    
    # h_ax.tick_params(axis="y", labelsize = 20)

    # h_ax.tick_params(axis="x", labelsize = 20)

    ax.tick_params(axis="x", labelsize = 20)

    ax.tick_params(axis="y", labelsize = 20)

    # h_ax.set_title('No-failure prediction', fontsize = 20, loc= 'center')

    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])

    ax.legend(ncol = 2, loc = 'center left', bbox_to_anchor=(-0.05, 1.1), fontsize = 20)
    # h_ax.legend(ncol = 2, loc = 'lower right', bbox_to_anchor=(1.0, 0), fontsize = 20)

    plt.tight_layout(rect=[0, 0, 1, 1])


    plt.savefig(plot_name)


    print("saved file:", plot_name)