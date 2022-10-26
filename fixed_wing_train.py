import os
import sys
import torch
import numpy as np
from dynamics.fixed_wing import FixedWing
from qp_control import config
from qp_control.constraints_fw import constraints
from qp_control.datagen import Dataset_with_Grad
from qp_control.trainer_new import Trainer
from qp_control.utils import Utils
from qp_control.NNfuncgrad import CBF, alpha_param, NNController_new
from pytictoc import TicToc

sys.path.insert(1, os.path.abspath('.'))

# import cProfile
# cProfile.run('foo()')


xg = torch.tensor([[150.0,
                    0.2,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0]])

x0 = torch.tensor([[120.0,
                    0.1,
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

# us_in = input("Training with fault (1) or no fault (0):")
# fault = int(us_in)

nominal_params = config.FIXED_WING_PARAMS

fault = nominal_params["fault"]

fault_control_index = 1

n_sample = 1000

t = TicToc()


def main():
    dynamics = FixedWing(x=x0, nominal_params=nominal_params, dt=dt, controller_dt=dt)
    util = Utils(n_state=n_state, m_control=m_control, j_const=2, dyn=dynamics, dt=dt, params=nominal_params,
                 fault=fault,
                 fault_control_index=fault_control_index)
    nn_controller = NNController_new(n_state=n_state, m_control=m_control)
    cbf = CBF(dynamics, n_state=n_state, m_control=m_control)
    alpha = alpha_param(n_state=n_state)

    try:
        if fault == 0:
            cbf.load_state_dict(torch.load('./data/FW_cbf_NN_weights.pth'))
            nn_controller.load_state_dict(torch.load('./data/FW_controller_NN_weights.pth'))
            alpha.load_state_dict(torch.load('./data/FW_alpha_NN_weights.pth'))
            cbf.eval()
            nn_controller.eval()
            alpha.eval()
        else:
            cbf.load_state_dict(torch.load('./data/FW_cbf_FT_weights.pth'))
            nn_controller.load_state_dict(torch.load('./data/FW_controller_FT_weights.pth'))
            alpha.load_state_dict(torch.load('./data/FW_alpha_FT_weights.pth'))
            cbf.eval()
            nn_controller.eval()
            alpha.eval()
    except:
        print("No pre-train data available")

    dataset = Dataset_with_Grad(n_state=n_state, m_control=m_control)
    trainer = Trainer(nn_controller, cbf, alpha, dataset, n_state=n_state, m_control=m_control, j_const=2, dyn=dynamics,
                      n_pos=1,
                      dt=dt, safe_alpha=0.3, dang_alpha=0.4, action_loss_weight=1, params=nominal_params,
                      fault=fault,
                      fault_control_index=fault_control_index)
    state = x0
    goal = xg
    goal = np.array(goal).reshape(1, n_state)

    safety_rate = 0.0
    goal_reached = 0.0
    um, ul = dynamics.control_limits()

    sm, sl = dynamics.state_limits()

    safe_m, safe_l = dynamics.safe_limits(sm, sl)

    u_nominal = torch.zeros(1, m_control)

    for i in range(config.TRAIN_STEPS):
        # t.tic()
        # print(i)
        if np.mod(i, config.INIT_STATE_UPDATE) == 0 and i > 0:
            init_states = util.x_bndr(safe_m, safe_l, n_sample)
            init_states = init_states.reshape(n_sample, n_state) + 10 * torch.randn(n_sample, n_state)
            init_u_nominal = torch.zeros(n_sample, m_control)
            init_u = util.nominal_controller(init_states, goal, init_u_nominal, dyn=dynamics,
                                             constraints=constraints)
            init_u = init_u.reshape(n_sample, m_control)
            init_unn = nn_controller(torch.tensor(init_states, dtype=torch.float32),
                                     torch.tensor(init_u, dtype=torch.float32))
            unn = torch.tensor(init_unn).reshape(n_sample, m_control)
            for j in range(n_sample):
                dataset.add_data(init_states[j, :], unn[j, :], init_u[j, :])

            # state = sl.clone().reshape(1, n_state) + torch.randn(1, n_state) * 10
            state = x0 + torch.randn(1, n_state) * 20
            state[0, 0] = safe_l[0] + 10 * torch.randn(1)
            state[0, 1] = safe_l[1] + 2 * torch.randn(1)
            state[0, 2] = safe_l[2] + 2 * torch.randn(1)

        if np.mod(i, 2 * config.INIT_STATE_UPDATE) == 0 and i > 0:
            # state = sm.clone().reshape(1, n_state) + torch.randn(1, n_state) * 10
            state = x0 + torch.randn(1, n_state) * 20
            state[0, 0] = safe_m[0] + 10 * torch.randn(1)
            state[0, 1] = safe_m[1] + 2 * torch.randn(1)
            state[0, 2] = safe_m[2] + 2 * torch.randn(1)

        # h, grad_h = cbf.V_with_jacobian(state.reshape(1, n_state, 1))
        # print(t.toc())

        for j in range(n_state):
            if state[0, j] < sl[j] * 0.5:
                state[0, j] = sl[j].clone()
            if state[0, j] > sm[j] * 2:
                state[0, j] = sm[j].clone()

        fx = dynamics._f(state, params=nominal_params)
        gx = dynamics._g(state, params=nominal_params)

        u_n = util.nominal_controller(state=state, goal=goal, u_n=u_nominal, dyn=dynamics, constraints=constraints)
        # u_nominal = util.neural_controller(u_n, fx, gx, h, grad_h, fault_start=0)

        u_nominal = u_n.reshape(1, m_control)
        # print("time till here: ")
        # time_taken = t.tocvalue()
        # time_taken = torch.tensor(time_taken, dtype=torch.float32)
        # print(time_taken)

        for j in range(m_control):
            if u_nominal[0, j] < ul[j]:
                u_nominal[0, j] = ul[j].clone()
            if u_nominal[0, j] > um[j]:
                u_nominal[0, j] = um[j].clone()

        u = nn_controller(torch.tensor(state, dtype=torch.float32), torch.tensor(u_nominal, dtype=torch.float32))

        u = torch.tensor(u).reshape(1, m_control)
        # u = torch.squeeze(u.detach())

        if torch.isnan(torch.sum(u)):
            u = (ul.clone().reshape(m_control) + um.clone().reshape(m_control)) / 2

        if fault == 1:
            u[0, fault_control_index] = torch.rand(1)

        for j in range(m_control):
            if u[0, j] < ul[j]:
                u[0, j] = ul[j].clone()
            if u[0, j] > um[j]:
                u[0, j] = um[j].clone()

        gxu = torch.matmul(gx, u.reshape(m_control, 1))

        dx = fx.reshape(1, n_state) + gxu.reshape(1, n_state)

        state_next = state + dx * dt

        dataset.add_data(state, u, u_nominal)

        is_safe = int(util.is_safe(state))

        safety_rate = safety_rate * (1 - 1 / config.POLICY_UPDATE_INTERVAL) + is_safe / config.POLICY_UPDATE_INTERVAL

        state = state_next
        # done = torch.linalg.norm(state_next.detach().cpu() - goal) < 5
        done = int(dynamics.goal_mask(state_next))
        if np.mod(i, config.POLICY_UPDATE_INTERVAL) == 0 and i > 0:
            loss_np, acc_np, loss_h_safe, loss_h_dang, loss_alpha, loss_deriv_safe, loss_deriv_dang, loss_deriv_mid, loss_action, loss_limit = trainer.train_cbf_and_controller()
            print(
                'step: {}, train h and u, loss: {:.3f}, safety rate: {:.3f}, goal reached: {:.3f}, acc: {}, '
                'loss_h_safe: {:.3f}, loss_h_dang: {:.3f}, loss_alpha: {:.3f}, loss_deriv_safe: {:.3f}, '
                'loss_deriv_dang: {:.3f}, loss_deriv_mid: {:.3f}, loss_action: {:.3f}, loss_limit: {:.3f}'.format(
                    i, loss_np, safety_rate, goal_reached, acc_np, loss_h_safe, loss_h_dang, loss_alpha,
                    loss_deriv_safe, loss_deriv_dang, loss_deriv_mid, loss_action, loss_limit))

            if fault == 0:
                torch.save(cbf.state_dict(), './data/FW_cbf_NN_weights.pth')
                torch.save(nn_controller.state_dict(), './data/FW_controller_NN_weights.pth')
                torch.save(alpha.state_dict(), './data/FW_alpha_NN_weights.pth')
            else:
                torch.save(cbf.state_dict(), './data/FW_cbf_FT_weights.pth')
                torch.save(nn_controller.state_dict(), './data/FW_controller_FT_weights.pth')
                torch.save(alpha.state_dict(), './data/FW_alpha_FT_weights.pth')
        if done:
            goal_reached = goal_reached * (1 - 1e-2) + done * 1e-2
            state = x0


if __name__ == '__main__':
    main()
