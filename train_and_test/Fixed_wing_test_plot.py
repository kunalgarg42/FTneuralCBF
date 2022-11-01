import os
import sys
import torch
import math
import random
import numpy as np
from dynamics.fixed_wing import FixedWing
from trainer import config
from trainer.constraints_fw import constraints
from trainer.datagen import Dataset_with_Grad
from trainer.trainer_fixed import Trainer
from trainer.utils import Utils
from trainer.NNfuncgrad import CBF, alpha_param, NNController_new
import matplotlib.pyplot as plt

sys.path.insert(1, os.path.abspath('..'))

xg = torch.tensor([[120.0,
                    0.2,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0]])

x0 = torch.tensor([[150.0,
                    0.2,
                    0.1,
                    0.4,
                    0.5,
                    0.2,
                    0.1,
                    0.5,
                    0.9]])

dt = 0.01
n_state = 9
m_control = 4

nominal_params = config.FIXED_WING_PARAMS

fault = nominal_params["fault"]

fault_control_index = 1
fault_duration = config.FAULT_DURATION

fault_known = 1


def main():
    dynamics = FixedWing(x=x0, nominal_params=nominal_params, dt=dt, controller_dt=dt)
    util = Utils(n_state=n_state, m_control=m_control, dyn=dynamics, params=nominal_params, fault=fault,
                 fault_control_index=fault_control_index)

    NN_controller = NNController_new(n_state=n_state, m_control=m_control)
    NN_cbf = CBF(dynamics, n_state=n_state, m_control=m_control, fault=0, fault_control_index=fault_control_index)
    # NN_alpha = alpha_param(n_state=n_state)

    NN_controller.load_state_dict(torch.load('./data/FW_controller_NN_weights.pth'))
    NN_cbf.load_state_dict(torch.load('./data/FW_cbf_NN_weights.pth'))
    # NN_alpha.load_state_dict(torch.load('./data/data/CF_alpha_NN_weights.pth'))

    NN_cbf.eval()
    NN_controller.eval()
    # NN_alpha.eval()

    FT_controller = NNController_new(n_state=n_state, m_control=m_control)
    FT_cbf = CBF(dynamics, n_state=n_state, m_control=m_control, fault=1, fault_control_index=fault_control_index)
    # FT_alpha = alpha_param(n_state=n_state)

    FT_controller.load_state_dict(torch.load('./data/FW_controller_FT_weights.pth'))
    FT_cbf.load_state_dict(torch.load('./data/FW_cbf_FT_weights.pth'))
    # FT_alpha.load_state_dict(torch.load('./data/data/CF_alpha_FT_weights.pth'))

    FT_cbf.eval()
    FT_controller.eval()
    # FT_alpha.eval()

    state = x0
    goal = xg
    goal = np.array(goal).reshape(1, n_state)

    safety_rate = 0.0
    unsafety_rate = 0.0
    h_correct = 0.0
    epsilon = 0.1

    um, ul = dynamics.control_limits()

    sm, sl = dynamics.state_limits()

    x_pl = np.array(state).reshape(1, n_state)
    fault_activity = np.array([0])
    u_pl = np.array([0] * m_control).reshape(1, m_control)
    h, _ = NN_cbf.V_with_jacobian(state.reshape(1, n_state, 1))

    # print(h)
    h_pl = np.array(h.detach()).reshape(1, 1)

    rand_start = random.uniform(1.01, 100)

    fault_start_epoch = math.floor(config.EVAL_STEPS / rand_start)  # + 100000000
    fault_start = 0
    u_nominal = torch.zeros(1, m_control)

    for i in range(config.EVAL_STEPS):
        # print(i)

        for j in range(n_state):
            if state[0, j] < sl[j]:
                state[0, j] = sl[j].clone()
            if state[0, j] > sm[j]:
                state[0, j] = sm[j].clone()

        fx = dynamics._f(state, params=nominal_params)
        gx = dynamics._g(state, params=nominal_params)

        u_nominal = util.nominal_controller(state=state, goal=goal, u_n=u_nominal, dyn=dynamics,
                                            constraints=constraints)
        # if fault_start == 0: u_nominal = NN_controller(torch.tensor(state, dtype=torch.float32), torch.tensor(
        # u_nominal, dtype=torch.float32)) else: u_nominal = FT_controller(torch.tensor(state, dtype=torch.float32),
        # torch.tensor(u_nominal, dtype=torch.float32))

        for j in range(m_control):
            if u_nominal[0, j] < ul[j]:
                u_nominal[0, j] = ul[j].clone()
            if u_nominal[0, j] > um[j]:
                u_nominal[0, j] = um[j].clone()

        if fault_known == 1:
            # 1 -> time-based switching, assumes knowledge of when fault occurs and stops
            # 0 -> Fault-detection based-switching, using the proposed scheme from the paper

            if fault_start == 0 and fault_start_epoch <= i <= (
                    fault_start_epoch + fault_duration):  # and util.is_safe(state):
                fault_start = 1

            # print(i)
            # print(fault_start_epoch + fault_duration)
            # print(fault_start)

            if fault_start == 1 and i > (fault_start_epoch + fault_duration):
                # print("here")
                fault_start = 0

            if fault_start == 0:
                h, grad_h = NN_cbf.V_with_jacobian(state.reshape(1, n_state, 1))
            else:
                h, grad_h = FT_cbf.V_with_jacobian(state.reshape(1, n_state, 1))

            u = util.neural_controller(u_nominal, fx, gx, h, grad_h, fault_start)
            u = torch.squeeze(u.detach().cpu())

            if fault_start == 1:
                u[fault_control_index] = torch.rand(1) * 5

            for j in range(m_control):
                if u[j] < ul[j]:
                    u[j] = ul[j].clone()
                if u[j] > um[j]:
                    u[j] = um[j].clone()

            # if torch.isnan(torch.sum(u)):
            # 	i = i-1
            # 	continue

            u = torch.tensor(u, dtype=torch.float32)
            gxu = torch.matmul(gx, u.reshape(m_control, 1))

            dx = fx.reshape(1, n_state) + gxu.reshape(1, n_state)

        else:
            h, grad_h = NN_cbf.V_with_jacobian(state.reshape(1, n_state, 1))
            u = util.neural_controller(u_nominal, fx, gx, h, grad_h, fault_start=fault_start)
            u = torch.squeeze(u.detach().cpu())

            if fault_start_epoch <= i <= fault_start_epoch + fault_duration:
                u[fault_control_index] = torch.rand(1)

            for j in range(m_control):
                if u[j] <= ul[j]:
                    u[j] = ul[j].clone()
                if u[j] >= um[j]:
                    u[j] = um[j].clone()

            if torch.isnan(torch.sum(u)):
                i = i - 1
                continue

            u = torch.tensor(u, dtype=torch.float32)
            gxu = torch.matmul(gx, u.reshape(m_control, 1))

            dx = fx.reshape(1, n_state) + gxu.reshape(1, n_state)

            dot_h = torch.matmul(dx, grad_h.reshape(n_state, 1))
            if dot_h < epsilon:
                h, grad_h = FT_cbf.V_with_jacobian(state.reshape(1, n_state, 1))
                u = util.neural_controller(u_nominal, fx, gx, h, grad_h, fault_start=fault_start)
                u = torch.squeeze(u.detach().cpu())
                if fault_start_epoch <= i <= fault_start_epoch + fault_duration:
                    u[fault_control_index] = torch.rand(1) * 5

                for j in range(m_control):
                    if u[j] < ul[j]:
                        u[j] = ul[j].clone()
                    if u[j] > um[j]:
                        u[j] = um[j].clone()

                if torch.isnan(torch.sum(u)):
                    i = i - 1
                    continue

                u = torch.tensor(u, dtype=torch.float32)
                gxu = torch.matmul(gx, u.reshape(m_control, 1))

                dx = fx.reshape(1, n_state) + gxu.reshape(1, n_state)

        state_next = state + dx * dt

        is_safe = int(util.is_safe(state))
        is_unsafe = int(util.is_unsafe(state))
        safety_rate += is_safe / config.EVAL_STEPS
        unsafety_rate += is_unsafe / config.EVAL_STEPS
        h_correct += is_safe * int(h >= 0) / config.EVAL_STEPS + is_unsafe * int(h < 0) / config.EVAL_STEPS

        x_pl = np.vstack((x_pl, np.array(state.clone().detach()).reshape(1, n_state)))
        fault_activity = np.vstack((fault_activity, fault_start))
        u_pl = np.vstack((u_pl, np.array(u.clone().detach()).reshape(1, m_control)))
        h_pl = np.vstack((h_pl, np.array(h.clone().detach()).reshape(1, 1)))

        state = state_next.clone()

    time_pl = np.arange(0., dt * config.EVAL_STEPS + dt, dt)

    alpha_pl = x_pl[:, 1]
    print(safety_rate)
    print(unsafety_rate)
    print(h_correct)

    u1 = u_pl[:, 0]
    u2 = u_pl[:, 1]
    u3 = u_pl[:, 2]
    u4 = u_pl[:, 3]

    plt.figure(figsize=(18, 6))

    plt.subplot(331)
    plt.plot(time_pl, alpha_pl)
    plt.subplot(334)
    plt.plot(time_pl, h_pl)
    plt.subplot(333)
    plt.plot(time_pl, u1, '--r')
    plt.subplot(332)
    plt.plot(time_pl, u2, '--g')
    plt.subplot(335)
    plt.plot(time_pl, u3, '--b')
    plt.subplot(336)
    plt.plot(time_pl, u4, '--y')
    # plt.suptitle('Categorical Plotting')
    plt.subplot(337)
    plt.plot(time_pl, x_pl[:, 0], '--r')
    plt.subplot(338)
    plt.plot(time_pl, x_pl[:, 7], '--b')
    plt.subplot(339)
    plt.plot(time_pl, fault_activity, '--g')
    plt.savefig('./plots/plot_closed_loop_data_FW.png')


if __name__ == '__main__':
    main()

# scp -r kgarg@18.18.47.27:/home/kgarg/kunal_files/MIT_REALM/fault_tol_control/data/data /home/kunal/MIT_REALM/Research/fault_tol_control/data/
