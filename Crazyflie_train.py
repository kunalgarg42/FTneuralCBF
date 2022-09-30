import os
import sys

sys.path.insert(1, os.path.abspath('.'))

import torch
import numpy as np
# from dynamics.fixed_wing_dyn import fw_dyn_ext, fw_dyn
from dynamics.Crazyflie import CrazyFlies
from qp_control import config
from qp_control.constraints_crazy import constraints

from qp_control.datagen_CF import Dataset_with_Grad
from qp_control.trainer_crazy import Trainer
from qp_control.utils_crazy import Utils
from qp_control.NNfuncgrad_CF import CBF, alpha_param, NNController_new

xg = torch.tensor([[0.0,
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
                    0.0]])

x0 = torch.tensor([[2.0,
                    2.0,
                    3.1,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0]])

dt = 0.01
n_state = 12
m_control = 4

us_in = input("Training with fault (1) and without fault (0):")

fault = int(us_in)

nominal_params = {
    "m": 0.0299,
    "Ixx": 1.395 * 10**(-5),
    "Iyy": 1.395 * 10**(-5),
    "Izz": 2.173 * 10**(-5),
    "CT": 3.1582 * 10**(-10),
    "CD": 7.9379 * 10**(-12),
    "d": 0.03973,
    "fault": fault,}

state = []
goal = []

fault_control_index = 1


def main():
    dynamics = CrazyFlies(x=x0, nominal_params=nominal_params, dt=dt, controller_dt=dt)
    util = Utils(n_state=n_state, m_control = m_control, dyn = dynamics, params = nominal_params, fault = fault, fault_control_index = fault_control_index)
    nn_controller = NNController_new(n_state=n_state, m_control=m_control)
    cbf = CBF(dynamics = dynamics, n_state=n_state, m_control=m_control,fault = fault, fault_control_index = fault_control_index)

    try:
        if fault == 0:
            # cbf = CBF(dynamics, n_state=n_state, m_control=m_control)
            cbf.load_state_dict(torch.load('./data/CF_cbf_NN_weights.pth'))
            cbf.eval()
        else:
            # cbf = CBF(dynamics, n_state=n_state, m_control=m_control)
            cbf.load_state_dict(torch.load('./data/CF_cbf_FT_weights.pth'))
            cbf.eval()
    except:
        print("No pre-train data available")

    alpha = alpha_param(n_state=n_state)
    dataset = Dataset_with_Grad(n_state=n_state, m_control=m_control, n_pos=1, safe_alpha=0.3, dang_alpha=0.4)
    trainer = Trainer(nn_controller, cbf, alpha, dataset, n_state=n_state, m_control=m_control, j_const=2, dyn=dynamics, n_pos=1,
                      dt=dt, safe_alpha=0.3, dang_alpha=0.4, action_loss_weight=0.001, params=nominal_params, fault=fault,
                      fault_control_index=fault_control_index)
    state = x0
    goal = xg
    goal = np.array(goal).reshape(1, n_state)

    state_error = torch.zeros(1, n_state)

    safety_rate = 0.0
    goal_reached = 0.0
    loss_total = 100.0

    um, ul = dynamics.control_limits()

    sm, sl = dynamics.state_limits()

    for i in range(config.TRAIN_STEPS):
        if np.mod(i, 2 * config.INIT_STATE_UPDATE) == 0 and i > 0:
            # state = torch.tensor(goal.copy()).reshape(1,n_state) + 0.2 * torch.randn(1,n_state)
            state = xg + 0.2 * torch.randn(1,n_state)

        for j in range(n_state):
            if state[0, j] < sl[j]:
                state[0, j] = xg[0,j].clone()
            if state[0, j] > sm[j]:
                state[0, j] = xg[0,j].clone()

        fx = dynamics._f(state, params=nominal_params)
        gx = dynamics._g(state, params=nominal_params)

        u_nominal = util.nominal_controller(state=state, goal=goal, u_norm_max=5, dyn=dynamics,
                                               constraints=constraints)
        u_nominal = dynamics.u_eq()

        for j in range(m_control):
            if u_nominal[0, j] < ul[j]:
                u_nominal[0, j] = ul[j].clone()
            if u_nominal[0, j] > um[j]:
                u_nominal[0, j] = um[j].clone()

        u = nn_controller(torch.tensor(state, dtype=torch.float32), torch.tensor(u_nominal, dtype=torch.float32))

        u = torch.squeeze(u.detach().cpu())

        if fault == 1:
            u[fault_control_index] = torch.rand(1) / 4

        if torch.isnan(torch.sum(u)):
            i = i - 1
            continue

        gxu = torch.matmul(gx, u.reshape(m_control, 1))

        dx = fx.reshape(1, n_state) + gxu.reshape(1, n_state)

        state_next = state + dx * dt

        h, _ = cbf.V_with_jacobian(state.reshape(1, n_state, 1))

        dataset.add_data(state, u, u_nominal)

        is_safe = int(util.is_safe(state))
        safety_rate = safety_rate * (1 - 1 / config.POLICY_UPDATE_INTERVAL) + is_safe / config.POLICY_UPDATE_INTERVAL

        state = state_next.clone()
        goal_err = state_next.detach().cpu() - goal
        done = torch.linalg.norm(goal_err[0,0:2]) < 1

        if np.mod(i, config.POLICY_UPDATE_INTERVAL) == 0 and i > 0:
            loss_np, acc_np, loss_h_safe, loss_h_dang, loss_alpha, loss_deriv_safe , loss_deriv_dang , loss_deriv_mid , loss_action = trainer.train_cbf_and_controller()
            print('step: {}, train h and u, loss: {:.3f}, safety rate: {:.3f}, goal reached: {:.3f}, acc: {}, loss_h_safe: {:.3f}, loss_h_dang: {:.3f}, loss_alpha: {:.3f}, loss_deriv_safe: {:.3f}, loss_deriv_dang: {:.3f}, loss_deriv_mid: {:.3f}, loss_action: {:.3f}'.format(
                i, loss_np, safety_rate, goal_reached, acc_np, loss_h_safe, loss_h_dang, loss_alpha, loss_deriv_safe , loss_deriv_dang , loss_deriv_mid , loss_action))
            loss_total = loss_np

            if fault == 0:
                torch.save(cbf.state_dict(), './data/CF_cbf_NN_weights.pth')
                torch.save(nn_controller.state_dict(), './data/CF_controller_NN_weights.pth')
                torch.save(alpha.state_dict(), './data/CF_alpha_NN_weights.pth')
            else:
                torch.save(cbf.state_dict(), './data/CF_cbf_FT_weights.pth')
                torch.save(nn_controller.state_dict(), './data/CF_controller_FT_weights.pth')
                torch.save(alpha.state_dict(), './data/CF_alpha_FT_weights.pth')

        if done:
            dist = np.linalg.norm(np.array(state_next, dtype=float) - np.array(goal, dtype=float))
            goal_reached = goal_reached * (1 - 1e-2) + (dist < 2.0) * 1e-2
            state = sl + torch.rand(1,n_state) * 0.1


if __name__ == '__main__':
    main()
