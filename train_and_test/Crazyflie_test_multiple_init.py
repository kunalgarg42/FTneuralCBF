import os
import sys

import numpy
import torch
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import tqdm

import seaborn as sns

sns.set_theme(context="paper", font_scale=2.0, style="ticks")

sys.path.insert(1, os.path.abspath(".."))
sys.path.insert(1, os.path.abspath("."))

# from dynamics.fixed_wing_dyn import fw_dyn_ext, fw_dyn
from dynamics.Crazyflie import CrazyFlies
from trainer import config
from trainer.constraints_crazy import constraints
from trainer.datagen import Dataset_with_Grad
from trainer.trainer import Trainer
from trainer.utils import Utils
from trainer.NNfuncgrad_CF import CBF, alpha_param, NNController_new

xg = torch.tensor([0.0, 0.0, 3.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

x0 = torch.tensor([[2.0, 2.0, 5.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
dt = 0.001
n_state = 12
m_control = 4
fault = 0

nominal_params = config.CRAZYFLIE_PARAMS

fault_control_index = 1
fault_duration = config.FAULT_DURATION

fault_known = 0


def main():
    dynamics = CrazyFlies(x=x0, goal=xg, nominal_params=nominal_params, dt=dt)
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
            torch.load("./good_data/data/CF_cbf_NN_weightsCBF.pth")
        )
        FT_cbf.load_state_dict(
            torch.load("./good_data/data/CF_cbf_FT_weightsCBF.pth")
        )
        # NN_alpha.load_state_dict(torch.load('./good_data/data/CF_alpha_NN_weights.pth'))
    except:
        # NN_controller.load_state_dict(torch.load('./data/CF_controller_NN_weights.pth'))
        NN_cbf.load_state_dict(
            torch.load("./data/CF_cbf_NN_weightsCBF.pth")
        )
        FT_cbf.load_state_dict(
            torch.load("./data/CF_cbf_FT_weightsCBF.pth")
        )
        # NN_alpha.load_state_dict(torch.load('./data/CF_alpha_NN_weights.pth'))

    NN_cbf.eval()

    FT_cbf.eval()

    state = x0
    um, ul = dynamics.control_limits()
    um = um.reshape(1, m_control).repeat(1, 1)
    ul = ul.reshape(1, m_control).repeat(1, 1)

    um = um.type(torch.FloatTensor)
    ul = ul.type(torch.FloatTensor)

    sm, sl = dynamics.state_limits()

    safety_rate_pl = np.array([1])
    unsafety_rate_pl = np.array([0])
    h_correct_pl = np.array([0])
    dot_h_pl = np.array([0])
    iteration = 1
    for iteration in tqdm.trange(100):

        fault_start = 0
        previous_state = state.clone()

        safety_rate = 0.0
        unsafety_rate = 0.0
        h_correct = 0.0
        dot_h_correct = 0.0
        epsilon = 0.1 - iteration / 1000

        state = x0.clone() + torch.randn(1, n_state) / 10

        rand_start = random.uniform(1.01, 100)

        fault_start_epoch = 5 * math.floor(config.EVAL_STEPS / rand_start)

        for i in range(config.EVAL_STEPS):
            # print(i)
            # if state[0, 2] > goal[0, 2]:
            #     u_nominal = u_eq.clone() * (1 + torch.linalg.norm(state[0, 2]-goal[0, 2]) / 1000)
            # else:
            #     u_nominal = u_eq.clone() * (1 - torch.linalg.norm(state[0, 2] - goal[0, 2]) / 1000)
            u_nominal = dynamics.u_nominal(state)

            for j in range(n_state):
                if state[0, j] < sl[j]:
                    state[0, j] = sl[j].clone()
                if state[0, j] > sm[j]:
                    state[0, j] = sm[j].clone()

            fx = dynamics._f(state, params=nominal_params)
            gx = dynamics._g(state, params=nominal_params)

            if fault_known == 1:
                # 1 -> time-based switching, assumes knowledge of when fault occurs and stops
                # 0 -> Fault-detection based-switching, using the proposed scheme from the paper

                if (
                        fault_start == 0
                        and fault_start_epoch <= i <= fault_start_epoch + fault_duration / 5
                        and util.is_safe(state)
                ):
                    fault_start = 1

                if fault_start == 1 and i > fault_start_epoch + fault_duration:
                    fault_start = 0

                if fault_start == 0:
                    h, grad_h = NN_cbf.V_with_jacobian(state.reshape(1, n_state, 1))
                else:
                    h, grad_h = FT_cbf.V_with_jacobian(state.reshape(1, n_state, 1))

                u = util.neural_controller(u_nominal, fx, gx, h, grad_h, fault_start)

                # u_nominal = util.neural_controller(u_nominal, fx, gx, h_nom, grad_h_nom, fault_start)
                # u_nominal = util.nominal_controller(state, goal, u_nominal, dynamics)

                # u = NN_controller(state, torch.tensor(u_nominal, dtype=torch.float32))
                u = u.reshape(1, m_control)

                if fault_start == 1:
                    u[0, fault_control_index] = torch.rand(1) / 4

                for j in range(m_control):
                    if u[0, j] < ul[0, j]:
                        u[0, j] = ul[0, j].clone() * 2
                    if u[0, j] > um[0, j]:
                        u[0, j] = um[0, j].clone() / 2

                u = torch.tensor(u, dtype=torch.float32)
                gxu = torch.matmul(gx, u.reshape(m_control, 1))

                dx = fx.reshape(1, n_state) + gxu.reshape(1, n_state)
                detect = fault_start

            else:
                h, grad_h = NN_cbf.V_with_jacobian(state.reshape(1, n_state, 1))
                h_prev, _ = NN_cbf.V_with_jacobian(previous_state.reshape(1, n_state, 1))
                # u = NN_controller(state, u_nominal)
                u = util.neural_controller(u_nominal, fx, gx, h, grad_h, fault_start)

                u = u.reshape(1, m_control)

                if fault_start_epoch <= i <= fault_start_epoch + fault_duration:
                    u[0, fault_control_index] = torch.rand(1) / 4
                    fault_start = 1.0
                else:
                    fault_start = 0.0

                for j in range(m_control):
                    if u[0, j] < ul[0, j]:
                        u[0, j] = ul[0, j].clone()
                    if u[0, j] > um[0, j]:
                        u[0, j] = um[0, j].clone()

                u = torch.tensor(u, dtype=torch.float32)
                gxu = torch.matmul(gx, u.reshape(m_control, 1))

                dx = fx.reshape(1, n_state) + gxu.reshape(1, n_state)

                # dot_h = torch.matmul(dx, grad_h.reshape(n_state, 1))
                dot_h = (h - h_prev) / dt + 1 * h

                # If no fault previously detected and dot_h is too small, then detect a fault
                if dot_h < epsilon - 100 * dt:
                    detect = 1
                    h, grad_h = FT_cbf.V_with_jacobian(state.reshape(1, n_state, 1))
                    u = util.neural_controller(u_nominal, fx, gx, h, grad_h, fault_start)
                    u  = u.reshape(1, m_control)
                    u = torch.tensor(u, dtype=torch.float32)

                    if fault_start_epoch <= i <= fault_start_epoch + fault_duration:
                        u[0, fault_control_index] = torch.rand(1) / 4

                    for j in range(m_control):
                        if u[0, j] <= ul[0, j]:
                            u[0, j] = ul[0, j].clone()
                        if u[0, j] >= um[0, j]:
                            u[0, j] = um[0, j].clone()

                    gxu = torch.matmul(gx, u.reshape(m_control, 1))

                    dx = fx.reshape(1, n_state) + gxu.reshape(1, n_state)
                # If we have previously detected a fault, switch to no fault if dot_h is
                # increasing
                # elif dot_h > 0 * epsilon / 10:
                else:
                    detect = 0

            # dot_h = torch.matmul(dx, grad_h.reshape(n_state, 1))
            # print(u)
            # u_nominal = u.clone().reshape(1, m_control)
            dot_h = util.doth_max_alpha(h, grad_h, fx, gx, um, ul)

            state_next = state + dx * dt

            previous_state = state.clone()

            is_safe = int(util.is_safe(state))
            is_unsafe = int(util.is_unsafe(state))
            safety_rate += is_safe / config.EVAL_STEPS

            unsafety_rate += is_unsafe / config.EVAL_STEPS
            h_correct += (
                    is_safe * int(h >= 0) / config.EVAL_STEPS
                    + is_unsafe * int(h < 0) / config.EVAL_STEPS
            )
            dot_h_correct += torch.sign(dot_h.clone().detach()) / config.EVAL_STEPS

            state = state_next.clone()
            # print('h, {}, dot_h, {}'.format(h.detach().cpu().numpy()[0][0], dot_h.detach().cpu().numpy()[0][0]))
        safety_rate_pl = np.vstack((safety_rate_pl, safety_rate))
        unsafety_rate_pl = np.vstack((unsafety_rate_pl, unsafety_rate))
        dot_h_pl = np.vstack((dot_h_pl, dot_h_correct))
        h_correct_pl = np.vstack((h_correct_pl, h_correct))

    time_pl = np.arange(0.1 + epsilon, epsilon, -1 / 1000)
    plot_len = safety_rate_pl.shape[0]
    # time_pl = time_pl[0:]
    colors = sns.color_palette()

    fig = plt.figure(figsize=(12, 14))
    fig.tight_layout(pad=1.15)
    axs = fig.subplots(2, 1)

    # Plot the altitude and CBF value on one axis
    z_ax = axs[0]
    z_ax.plot(time_pl[0:plot_len], safety_rate_pl[0:plot_len], linewidth=4.0, label="safety rate", color=colors[0])
    z_ax.plot(time_pl[0:plot_len], unsafety_rate_pl[0:plot_len], linewidth=4.0, label="un safety rate", color=colors[1])

    z_ax.set_ylabel("Safety and unsafety rate", color=colors[0])
    z_ax.set_xlabel("Iteration")
    z_ax.set_xlim(time_pl[1], time_pl[-1])
    z_ax.tick_params(axis="y", labelcolor=colors[0])
    z_ax.legend()

    # Plot the control action on another axis
    u_ax = axs[1]
    u_ax.plot(time_pl[0:plot_len], dot_h_pl[0:plot_len], linewidth=2.0, label="$\dot h > 0$")
    u_ax.plot(time_pl[0:plot_len], h_correct_pl[0:plot_len], linewidth=2.0, label="$h$")

    u_ax.set_xlabel("Iteration")
    u_ax.set_ylabel("$h$ and $\dot h > 0$")
    u_ax.set_xlim(time_pl[1], time_pl[-1])
    u_ax.legend()

    # Add the fault indicators

    plt.savefig("./plots/plot_CF_iter.png")


if __name__ == "__main__":
    main()
