import os
import sys

import numpy
import torch
import math
import random
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(1, os.path.abspath('..'))
sys.path.insert(1, os.path.abspath('.'))

# from dynamics.fixed_wing_dyn import fw_dyn_ext, fw_dyn
from dynamics.Crazyflie import CrazyFlies
from trainer import config
from trainer.constraints_crazy import constraints
from trainer.datagen import Dataset_with_Grad
from trainer.trainer import Trainer
from trainer.utils import Utils
from trainer.NNfuncgrad_CF import CBF, alpha_param, NNController_new

xg = torch.tensor([0.0,
                   0.0,
                   3.5,
                   0.0,
                   0.0,
                   0.0,
                   0.0,
                   0.0,
                   0.0,
                   0.0,
                   0.0,
                   0.0])

x0 = torch.tensor([[2.0,
                    2.0,
                    5.1,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0]])
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
    util = Utils(n_state=n_state, m_control=m_control, dyn=dynamics, params=nominal_params, fault=fault,
                 fault_control_index=fault_control_index, j_const=2)

    # NN_controller = NNController_new(n_state=n_state, m_control=m_control)
    NN_cbf = CBF(dynamics, n_state=n_state, m_control=m_control)
    # NN_alpha = alpha_param(n_state=n_state)
    FT_cbf = CBF(dynamics, n_state=n_state, m_control=m_control)
    try:
        # NN_controller.load_state_dict(torch.load('./good_data/data/CF_controller_NN_weightsCBF.pth'))
        NN_cbf.load_state_dict(torch.load('./good_data/data/CF_cbf_NN_weightsCBF.pth', map_location='cpu'))
        FT_cbf.load_state_dict(torch.load('./good_data/data/CF_cbf_FT_weightsCBF.pth', map_location='cpu'))
        # NN_alpha.load_state_dict(torch.load('./good_data/data/CF_alpha_NN_weights.pth'))
    except:
        # NN_controller.load_state_dict(torch.load('./data/CF_controller_NN_weights.pth'))
        NN_cbf.load_state_dict(torch.load('./data/CF_cbf_NN_weightsCBF.pth', map_location='cpu'))
        FT_cbf.load_state_dict(torch.load('./data/CF_cbf_FT_weightsCBF.pth', map_location='cpu'))
        # NN_alpha.load_state_dict(torch.load('./data/CF_alpha_NN_weights.pth'))

    NN_cbf.eval()
    # NN_controller.eval()
    # NN_alpha.eval()

    # FT_controller = NNController_new(n_state=n_state, m_control=m_control)

    # FT_alpha = alpha_param(n_state=n_state)

    # FT_controller.load_state_dict(torch.load('./good_data/data/CF_controller_FT_weights.pth'))
    # FT_alpha.load_state_dict(torch.load('./good_data/data/CF_alpha_FT_weights.pth'))

    FT_cbf.eval()
    # FT_controller.eval()
    # FT_alpha.eval()

    state = x0

    safety_rate = 0.0
    unsafety_rate = 0.0
    h_correct = 0.0
    dot_h_correct = 0.0
    epsilon = 0.1

    um, ul = dynamics.control_limits()
    um = um.reshape(1, m_control).repeat(1, 1)
    ul = ul.reshape(1, m_control).repeat(1, 1)

    um = um.type(torch.FloatTensor)
    ul = ul.type(torch.FloatTensor)

    sm, sl = dynamics.state_limits()

    x_pl = np.array(state).reshape(1, n_state)
    fault_activity = np.array([0])
    detect_activity = np.array([0])

    u_pl = np.array([0] * m_control).reshape(1, m_control)
    h, _ = NN_cbf.V_with_jacobian(state.reshape(1, n_state, 1))

    # print(h)
    h_pl = np.array(h.detach()).reshape(1, 1)

    rand_start = random.uniform(1.01, 100)

    fault_start_epoch = math.floor(config.EVAL_STEPS / rand_start)
    fault_start = 0
    detect = 0
    # u_nominal = 0.05 * torch.ones(1, m_control)

    # u_samples = numpy.linspace(ul, um, num=10000)
    #
    # u_samples = u_samples.reshape(10000, 4)
    # alpha_samples = numpy.linspace(-10, 10, num=10000)
    # alpha_samples = alpha_samples.reshape(10000, 1)
    # u_samples = np.hstack((u_samples, alpha_samples))
    # u_samples = torch.tensor(u_samples, dtype=torch.float32)
    previous_state = state.clone()

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

            if fault_start == 0 and fault_start_epoch <= i <= fault_start_epoch + fault_duration / 5 and util.is_safe(
                    state):
                fault_start = 1

            if fault_start == 1 and i > fault_start_epoch + fault_duration:
                fault_start = 0

            if fault_start == 0:
                h, grad_h = NN_cbf.V_with_jacobian(state.reshape(1, n_state, 1))
            else:
                h, grad_h = FT_cbf.V_with_jacobian(state.reshape(1, n_state, 1))

            # h_nom = - torch.sum((state - (safe_m + safe_l) / 2) ** 2 + ((safe_m - safe_l) / 2) ** 2)
            # h_nom = h_nom.reshape(1, 1)
            # grad_h_nom = - 2 * (state - (sm + sl) / 2).reshape(1, 1, n_state)
            # LgH = torch.matmul(grad_h, gx).reshape(m_control, 1)
            # LgH = torch.vstack((LgH, h.reshape(1, 1)))
            # u = ul[0, 0].clone() * torch.ones(1, m_control)
            # for j in range(m_control):
            #     dotLghu = u_samples[:, j] * LgH[j]
            #     try:
            #         ind = int(torch.argmax(dotLghu))
            #         u[0, j] = u_samples[ind, j]
            #     except:
            #         u[0, j] = ul[0, j].clone()
            # try:
            #     ind = int(ind[:, 0].min())
            #     u = u_samples[ind, 0:m_control]
            # except:
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

        else:
            h, grad_h = NN_cbf.V_with_jacobian(state.reshape(1, n_state, 1))
            h_prev, _ = NN_cbf.V_with_jacobian(previous_state.reshape(1, n_state, 1))
            # u = NN_controller(state, u_nominal)
            u = util.neural_controller(u_nominal, fx, gx, h, grad_h, fault_start)

            if fault_start_epoch <= i <= fault_start_epoch + fault_duration:
                u[0, fault_control_index] = torch.rand(1) / 4

            for j in range(m_control):
                if u[0, j] < ul[0, j]:
                    u[0, j] = ul[0, j].clone()
                if u[0, j] > um[0, j]:
                    u[0, j] = um[0, j].clone()

            u = torch.tensor(u, dtype=torch.float32)
            gxu = torch.matmul(gx, u.reshape(m_control, 1))

            dx = fx.reshape(1, n_state) + gxu.reshape(1, n_state)

            # dot_h = torch.matmul(dx, grad_h.reshape(n_state, 1))
            dot_h = (h - h_prev) / dt + 10 * h
            if dot_h < epsilon - 10 * dt:
                detect = 1
                h, grad_h = FT_cbf.V_with_jacobian(state.reshape(1, n_state, 1))
                # u = FT_controller(state, u_nominal)
                u = util.neural_controller(u_nominal, fx, gx, h, grad_h, fault_start)
                u = torch.tensor(u, dtype=torch.float32)
                # u = torch.squeeze(u.detach().cpu())
                if fault_start_epoch <= i <= fault_start_epoch + fault_duration:
                    u[0, fault_control_index] = torch.rand(1) / 4

                for j in range(m_control):
                    if u[0, j] <= ul[0, j]:
                        u[0, j] = ul[0, j].clone()
                    if u[0, j] >= um[0, j]:
                        u[0, j] = um[0, j].clone()

                u = torch.tensor(u, dtype=torch.float32)
                gxu = torch.matmul(gx, u.reshape(m_control, 1))

                dx = fx.reshape(1, n_state) + gxu.reshape(1, n_state)
            else:
                if detect == 1 and dot_h > epsilon:
                    detect = 0
                else:
                    detect = 1

        # dot_h = torch.matmul(dx, grad_h.reshape(n_state, 1))
        # print(u)
        # u_nominal = u.clone().reshape(1, m_control)
        detect_activity = np.vstack((detect_activity, detect))
        dot_h = util.doth_max_alpha(h, grad_h, fx, gx, um, ul)
        if dot_h < 0:
            print(i)
        state_next = state + dx * dt

        previous_state = state.clone()

        is_safe = int(util.is_safe(state))
        is_unsafe = int(util.is_unsafe(state))
        safety_rate = safety_rate * (1 - 1 / config.EVAL_STEPS) + is_safe / config.EVAL_STEPS
        unsafety_rate += is_unsafe / config.EVAL_STEPS
        h_correct += is_safe * int(h >= 0) / config.EVAL_STEPS + is_unsafe * int(h < 0) / config.EVAL_STEPS
        dot_h_correct += torch.sign(dot_h.clone().detach()) / config.EVAL_STEPS

        x_pl = np.vstack((x_pl, np.array(state.clone().detach()).reshape(1, n_state)))
        fault_activity = np.vstack((fault_activity, fault_start))
        u_pl = np.vstack((u_pl, np.array(u.clone().detach()).reshape(1, m_control)))
        h_pl = np.vstack((h_pl, np.array(h.clone().detach()).reshape(1, 1)))

        state = state_next.clone()
        # print('h, {}, dot_h, {}'.format(h.detach().cpu().numpy()[0][0], dot_h.detach().cpu().numpy()[0][0]))
    time_pl = np.arange(0., dt * config.EVAL_STEPS + dt, dt)

    z_pl = x_pl[:, 2]

    p_pl = x_pl[:, 4]

    ps_pl = x_pl[:, 3]

    print(safety_rate)
    print(unsafety_rate)
    print(h_correct)
    print(dot_h_correct)

    u1 = u_pl[:, 0]
    u2 = u_pl[:, 1]
    u3 = u_pl[:, 2]
    u4 = u_pl[:, 3]

    fig = plt.figure(figsize=(18, 10))
    fig.tight_layout(pad=5.0)

    ax1 = plt.subplot(331)
    ax1.plot(time_pl, z_pl)
    ax1.title.set_text('z')
    ax2 = plt.subplot(332)
    ax2.plot(time_pl, u2, '--g')
    ax2.title.set_text('u2')
    ax3 = plt.subplot(333)
    ax3.plot(time_pl, u1, '--r')
    ax3.title.set_text('u1')
    ax4 = plt.subplot(334)
    ax4.plot(time_pl, h_pl)
    ax4.title.set_text('CBF value')
    ax5 = plt.subplot(335)
    ax5.plot(time_pl, u3, '--b')
    ax5.title.set_text('u3')
    ax6 = plt.subplot(336)
    ax6.plot(time_pl, u4, '--y')
    ax6.title.set_text('u4')
    ax7 = plt.subplot(337)
    ax7.plot(time_pl, fault_activity, '--g')
    ax7.title.set_text('fault and detection activity')
    # ax7 = plt.subplot(337)
    ax7.plot(time_pl, detect_activity, '--r')
    # ax72.title.set_text('d')

    ax8 = plt.subplot(338)
    ax8.plot(time_pl, p_pl, '--g')
    ax8.title.set_text('Angle theta')
    ax9 = plt.subplot(339)
    ax9.plot(time_pl, ps_pl, '--r')
    ax9.title.set_text('Angle phi')
    # plt.suptitle('Categorical Plotting')
    plt.savefig('./plots/plot_closed_loop_data_CF.png')


if __name__ == '__main__':
    main()
