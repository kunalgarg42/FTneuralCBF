import os
import sys

import torch
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import tqdm

import seaborn as sns

sns.set_theme(context="paper", font_scale=3.0, style="ticks")

sys.path.insert(1, os.path.abspath(".."))
sys.path.insert(1, os.path.abspath("."))

# from dynamics.fixed_wing_dyn import fw_dyn_ext, fw_dyn
from dynamics.DI_dyn import DI
from trainer import config
from trainer.utils import Utils
from trainer.NNfuncgrad_CF import CBF

m_control = 3

n_state = m_control * 2

x0 = torch.randn(1, n_state)

xg = torch.randn(1, n_state)

dt = 0.001

fault = 0

nominal_params = config.CRAZYFLIE_PARAMS

fault_control_index = 1
fault_duration = config.FAULT_DURATION

fault_known = 1


def main():
    dynamics = DI(x=x0, goal=xg, dt=dt, dim=m_control, nominal_parameters=[])
    util = Utils(
        n_state=n_state,
        m_control=m_control,
        dyn=dynamics,
        params=nominal_params,
        fault=fault,
        fault_control_index=fault_control_index,
        j_const=2,
    )

    # NN_controller = NNController_new(n_state=n_state, m_control=m_control)
    NN_cbf = CBF(dynamics, n_state=n_state, m_control=m_control)
    # NN_alpha = alpha_param(n_state=n_state)
    FT_cbf = CBF(dynamics, n_state=n_state, m_control=m_control)
    try:
        # NN_controller.load_state_dict(torch.load('./good_data/data/CF_controller_NN_weightsCBF.pth'))
        NN_cbf.load_state_dict(
            torch.load(
                "./good_data/data/DI_cbf_NN_weightsCBF.pth",
                map_location=torch.device("cpu"),
            )
        )
        FT_cbf.load_state_dict(
            torch.load(
                "./good_data/data/DI_cbf_FT_weightsCBF.pth",
                map_location=torch.device("cpu"),
            )
        )
        # NN_alpha.load_state_dict(torch.load('./good_data/data/CF_alpha_NN_weights.pth'))
    except:
        # NN_controller.load_state_dict(torch.load('./data/CF_controller_NN_weights.pth'))
        NN_cbf.load_state_dict(
            torch.load(
                "./data/DI_cbf_NN_weightsCBF.pth", map_location=torch.device("cpu")
            )
        )
        FT_cbf.load_state_dict(
            torch.load(
                "./data/DI_cbf_FT_weightsCBF.pth", map_location=torch.device("cpu")
            )
        )
        # NN_alpha.load_state_dict(torch.load('./data/CF_alpha_NN_weights.pth'))

    NN_cbf.eval()

    FT_cbf.eval()

    state = x0.clone().reshape(1, n_state)
    state = state.repeat(4, 1)
    state_next = state.clone()

    safety_rate = np.array([0.0] * 4).reshape(4,)
    unsafety_rate = np.array([0.0] * 4).reshape(4,)
    h_correct = np.array([0.0] * 4).reshape(4,)
    dot_h_correct = np.array([0.0] * 4).reshape(4,)
    epsilon = 0.1

    um, ul = dynamics.control_limits()
    um = um.reshape(1, m_control).repeat(1, 1)
    ul = ul.reshape(1, m_control).repeat(1, 1)

    um = um.type(torch.FloatTensor)
    ul = ul.type(torch.FloatTensor)

    sm, sl = dynamics.state_limits()

    x_pl = torch.zeros(4, n_state, config.EVAL_STEPS)

    fault_activity = np.array([0])
    detect_activity = np.array([0]*4*config.EVAL_STEPS).reshape(4, config.EVAL_STEPS)

    u_pl = torch.zeros(4, m_control, config.EVAL_STEPS)
    h, _ = NN_cbf.V_with_jacobian(state.reshape(4, n_state, 1))

    h_pl = torch.zeros(4, config.EVAL_STEPS)

    rand_start = random.uniform(1.01, 50)

    fault_start_epoch = 10 * math.floor(config.EVAL_STEPS / rand_start)
    
    fault_start = 0
    
    detect = np.array([0]*4).reshape(4,)

    dot_h_pl = np.array([0]*4*config.EVAL_STEPS).reshape(4, config.EVAL_STEPS)

    previous_state = state.clone()

    for i in tqdm.trange(config.EVAL_STEPS):
        u_temp = torch.zeros(4, m_control)
        h_temp = torch.zeros(4, 1)
        for k in range(4):
            u_nominal = dynamics.u_nominal(state[k, :].reshape(1, n_state))

            for j in range(n_state):
                if state[k, j] < sl[j]:
                    state[k, j] = sl[j].clone()
                if state[k, j] > sm[j]:
                    state[k, j] = sm[j].clone()

            fx = dynamics._f(state[k, :].reshape(1, n_state), params=nominal_params)
            gx = dynamics._g(state[k, :].reshape(1, n_state), params=nominal_params)

            if fault_known == 1:
                # 1 -> time-based switching, assumes knowledge of when fault occurs and stops
                # 0 -> Fault-detection based-switching, using the proposed scheme from the paper

                if (
                    fault_start == 0
                    and fault_start_epoch <= i <= fault_start_epoch + fault_duration / 5
                    # and util.is_safe(state[k, :].reshape(1, n_state))
                ):
                    fault_start = 1

                if fault_start == 1 and i > fault_start_epoch + fault_duration:
                    fault_start = 0

                if fault_start == 0:
                    h, grad_h = NN_cbf.V_with_jacobian(state[k, :].reshape(1, n_state, 1))
                else:
                    h, grad_h = FT_cbf.V_with_jacobian(state[k, :].reshape(1, n_state, 1))

                u = util.neural_controller(u_nominal, fx, gx, h, grad_h, fault_start)
                
                u = u.reshape(1, m_control)

                if fault_start == 1:
                    if k == 0:
                        u[0, fault_control_index] = ul[0, 0].clone() * 20 # torch.rand(1) / 4
                    elif k == 1:
                        u[0, fault_control_index] = um[0, 0].clone()
                    elif k == 2:
                        u[0, fault_control_index] = (torch.sin(torch.tensor(i / 100)) ** 2) * um[0, 0].clone()
                    else:
                        u[0, fault_control_index] = torch.rand(1) / 4
                    
                for j in range(m_control):
                    if u[0, j] < ul[0, j]:
                        u[0, j] = ul[0, j].clone()
                    if u[0, j] > um[0, j]:
                        u[0, j] = um[0, j].clone()

                u = torch.tensor(u, dtype=torch.float32)
                gxu = torch.matmul(gx, u.reshape(m_control, 1))

                dx = fx.reshape(1, n_state) + gxu.reshape(1, n_state)
                detect[k] = fault_start

            else:
                h, grad_h = NN_cbf.V_with_jacobian(state.reshape(1, n_state, 1))
                h_prev, _ = NN_cbf.V_with_jacobian(previous_state.reshape(1, n_state, 1))
                u = util.neural_controller(u_nominal, fx, gx, h, grad_h, fault_start)

                u = u.reshape(1, m_control)

                if fault_start_epoch <= i <= fault_start_epoch + fault_duration:
                    u[0, fault_control_index] = ul[0, 0].clone() #  0 * (torch.sin(torch.tensor(i / 100)) ** 2) * um[0, 0].clone()
                    fault_start = 1.0
                else:
                    fault_start = 0.0

                for j in range(m_control):
                    if u[0, j] < ul[0, j]:
                        u[0, j] = ul[0, j].clone()
                    if u[0, j] > um[0, j]:
                        u[0, j] = um[0, j].clone()

                u = u.clone().type(torch.float32)
                gxu = torch.matmul(gx, u.reshape(m_control, 1))

                dx = fx.reshape(1, n_state) + gxu.reshape(1, n_state)

                dot_h = (h - h_prev) / dt + 0.01 * h
                

                # If no fault previously detected and dot_h is too small, then detect a fault
                if detect == 0 and dot_h < epsilon - 10 * dt:
                    detect = 1
                    h, grad_h = FT_cbf.V_with_jacobian(state.reshape(1, n_state, 1))

                    u = util.neural_controller(u_nominal, fx, gx, h, grad_h, fault_start)

                    u = u.reshape(1, m_control)

                    u = torch.tensor(u, dtype=torch.float32)

                    if fault_start_epoch <= i <= fault_start_epoch + fault_duration:
                        u[0, fault_control_index] = ul[0, 0].clone() #  0 * (torch.sin(torch.tensor(i / 100)) ** 2) * um[0, 0].clone() #  torch.rand(1) / 4

                    for j in range(m_control):
                        if u[0, j] <= ul[0, j]:
                            u[0, j] = ul[0, j].clone()
                        if u[0, j] >= um[0, j]:
                            u[0, j] = um[0, j].clone()

                    gxu = torch.matmul(gx, u.reshape(m_control, 1))

                    dx = fx.reshape(1, n_state) + gxu.reshape(1, n_state)
                # If we have previously detected a fault, switch to no fault if dot_h is
                # increasing
                elif (detect == 1 and dot_h > epsilon / 10):
                    # else:
                    detect = 0

            detect_activity[k, i] = detect[k].copy()
            
            if fault_known == 0:
                dot_h_pl[k, i] = dot_h.clone().detach().numpy()
            
            dot_h = util.doth_max_alpha(h, grad_h, fx, gx, um, ul)

            if fault_known == 1:
                dot_h_pl[k, i] = dot_h.clone().detach().numpy()

            if dot_h < 0:
                print(i)
            state_next[k, :] = state[k, :] + dx * dt

            is_safe = int(util.is_safe(state[k, :].reshape(1, n_state)))
            is_unsafe = int(util.is_unsafe(state[k, :].reshape(1, n_state)))
            safety_rate[k] += is_safe / config.EVAL_STEPS

            unsafety_rate[k] += is_unsafe / config.EVAL_STEPS
            h_correct[k] += (
                is_safe * int(h >= 0) / config.EVAL_STEPS
                + is_unsafe * int(h < 0) / config.EVAL_STEPS
            )
            dot_h_correct[k] += torch.sign(dot_h.clone().detach()) / config.EVAL_STEPS
            u_temp[k, :] = u.clone().detach()
            h_temp[k] = h.clone().detach()

        u_pl[:, :, i] = u_temp.reshape(4, m_control)
        
        h_pl[:, i] = h_temp.reshape(4,)

        x_pl[:, :, i] = state.detach().cpu().reshape(4, n_state)

        fault_activity = np.vstack((fault_activity, fault_start))
            
        previous_state = state.clone()

        state = state_next.clone()

        # print('h, {}, dot_h, {}'.format(h.detach().cpu().numpy()[0][0], dot_h.detach().cpu().numpy()[0][0]))
    u_pl = np.array(u_pl)
    h_pl = np.array(h_pl)
    # x_pl = np.array(x_pl)

    time_pl = np.arange(dt, dt * config.EVAL_STEPS + dt, dt)

    z_pl = np.array(torch.transpose(x_pl[:, 2, :], 0, 1))

    # print(x_pl[:, :, -1])

    print(safety_rate)
    print(unsafety_rate)
    print(h_correct)
    print(dot_h_correct)

    u1 = u_pl[:, 0, :]
    u2 = u_pl[:, 1, :]
    u3 = u_pl[:, 2, :]

    linestyles = ["-", "--", "dashdot", ":"]

    colors = sns.color_palette()

    fig = plt.figure(figsize=(31, 9))
    axs = fig.subplots(2, 3)

    # Plot the altitude and CBF value on one axis
    z_ax = axs[0, 0]
    for k in range(4):
        if k == 0:
            z_ax.plot(time_pl, z_pl[:, k], linewidth=4.0, label="z (m)", color=colors[0], linestyle=linestyles[k])
        else:
            z_ax.plot(time_pl, z_pl[:, k], linewidth=4.0, color=colors[0], linestyle=linestyles[k])
    # z_ax.plot(time_pl, dot_h_pl, linewidth=2.0, label="z (m)", color=colors[2])

    unsafe_z = 1.0
    z_ax.plot(
        time_pl,
        0 * time_pl + unsafe_z,
        color="k",
        linestyle="--",
        linewidth=4.0,
    )
    z_ax.text(time_pl.max() * 0.05, unsafe_z + 0.1, "Unsafe boundary")
    z_ax.plot([], [], color=colors[1], linestyle="-", linewidth=4.0, label="CBF h(x)")
    z_ax.set_ylabel("Height (m)", color=colors[0])
    z_ax.set_xlabel("Time (s)")
    z_ax.set_xlim(time_pl[0], time_pl[-1] + 0.1)
    z_ax.tick_params(axis="y", labelcolor=colors[0])
    z_ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=2, frameon=False)

    h_ax = z_ax.twinx()

    for k in range(4):
        h_ax.plot(time_pl, h_pl[k, :].reshape(config.EVAL_STEPS, 1), linestyle=linestyles[k], linewidth=4.0, color=colors[1])
    
    h_ax.plot(
        time_pl,
        0 * time_pl,
        color="k",
        linestyle="--",
        linewidth=4.0,
        label="Unsafe boundary",
    )
    h_ax.set_ylabel("CBF value", color=colors[1])
    h_ax.tick_params(axis="y", labelcolor=colors[1])

    # Plot the control action on another axis
    mean_u = (u_pl.max() + u_pl.min()) / 2.0

    # Add the fault indicators
    fault_activity[-2] = 1.0
    fault_activity[-1] = 0.0
    (t_fault_start, t_fault_end) = time_pl[
        np.diff(fault_activity.squeeze()).nonzero()[0]
    ]

    for k in range(3):
        u_ax = axs[0 + np.mod(k, 2), 1 + 1 * (k > 1)]
        # if k == 0:
        u_ax.plot(time_pl, u2[k, :].reshape(config.EVAL_STEPS, 1), linewidth=2.0, label="$u_2$ (faulty)", linestyle=linestyles[k])
        u_ax.plot(time_pl, u1[k, :].reshape(config.EVAL_STEPS, 1), linewidth=2.0, label="$u_1$", linestyle=linestyles[k])
        u_ax.plot(time_pl, u3[k, :].reshape(config.EVAL_STEPS, 1), linewidth=2.0, label="$u_3$", linestyle=linestyles[k])
        # u_ax.plot(time_pl, u4[k, :].reshape(config.EVAL_STEPS, 1), linewidth=2.0, label="$u_4$", linestyle=linestyles[k])
        # else:
        #     u_ax.plot(time_pl, u2[k, :].reshape(config.EVAL_STEPS, 1), linewidth=2.0, linestyle=linestyles[k])
        #     u_ax.plot(time_pl, u1[k, :].reshape(config.EVAL_STEPS, 1), linewidth=2.0, linestyle=linestyles[k])
        #     u_ax.plot(time_pl, u3[k, :].reshape(config.EVAL_STEPS, 1), linewidth=2.0, linestyle=linestyles[k])
        #     u_ax.plot(time_pl, u4[k, :].reshape(config.EVAL_STEPS, 1), linewidth=2.0, linestyle=linestyles[k])
        u_ax.set_xlabel("Time (s)")
        u_ax.set_ylabel("Control effort")
        u_ax.set_xlim(time_pl[0], time_pl[-1] + 0.1)
        u_ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.17),
            ncol=4,
            frameon=False,
            columnspacing=0.7,
            handlelength=0.7,
            handletextpad=0.3,
        )
        lims = u_ax.get_ylim()
        u_ax.fill_between(
            [t_fault_start, t_fault_end],
            [-1.0, -1.0],
            [1.0, 1.0],
            color="grey",
            alpha=0.5,
            label="Fault",
        )
        
        u_ax.text(
            t_fault_start,
            mean_u,
            "Fault occurs",
            rotation="vertical",
            horizontalalignment="right",
            verticalalignment="center",
        )
        u_ax.text(
            t_fault_end,
            mean_u,
            "Fault clears",
            rotation="vertical",
            horizontalalignment="right",
            verticalalignment="center",
        )
        u_ax.set_ylim(lims)

    # Plot the fault detection on a third axis
    # w_ax = axs[2]
    dot_h_pl[0] = dot_h_pl[1]  # remove dummy value from start
    w_ax = axs[1, 0]
    for k in range(4):
        if k == 0:
            w_ax.plot(time_pl, dot_h_pl[k, :].reshape(config.EVAL_STEPS, 1), linewidth=4.0, label="$\omega$", linestyle=linestyles[k], color = colors[2])
        else:
            w_ax.plot(time_pl, dot_h_pl[k, :].reshape(config.EVAL_STEPS, 1), linewidth=4.0, linestyle=linestyles[k], color = colors[2])
    
    w_ax.plot(
        time_pl,
        0 * time_pl + epsilon - 10 * dt,
        "--",
        color="grey",
        # label="detection threshold",
    )

    w_ax.set_xlabel("Time (s)")
    
    lims = z_ax.get_ylim()
    fault_handle = z_ax.fill_between(
        [t_fault_start, t_fault_end],
        [-10.0, -10.0],
        [10.0, 10.0],
        color="grey",
        alpha=0.5,
        label="Fault",
    )
    z_ax.set_ylim(lims)
    lims = w_ax.get_ylim()
    w_ax.fill_between(
        [t_fault_start, t_fault_end],
        [-100.0, -100.0],
        [100.0, 100.0],
        color="grey",
        alpha=0.5,
    )
    w_ax.set_ylim(lims)
    for k in range(4):
        if k == 0:
            w_ax.plot(
                time_pl,
                detect_activity[k, :].reshape(config.EVAL_STEPS, 1),
                color=colors[3],
                label="Fault detected",
                linewidth=4.0,
                linestyle = linestyles[k],
            )
        else:
            w_ax.plot(
                time_pl,
                detect_activity[k, :].reshape(config.EVAL_STEPS, 1),
                color=colors[3],
                linewidth=4.0,
                linestyle = linestyles[k]
            )
    
    w_ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.17),
        ncol=3,
        frameon=False,
        # columnspacing=0.7,
        # handlelength=0.7,
        # handletextpad=0.3,
    )

    fig.tight_layout(pad=1.15)
    if fault_known == 1:
        plt.savefig("./plots/plot_DI_known_F_all.png")
    else:
        plt.savefig("./plots/plot_DI_all.png")


if __name__ == "__main__":
    main()
