import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
from qp_control import config
from qp_control.constraints_fw import constraints
from qp_control.NNfuncgrad import CBF, NNController_new, alpha_param
from dynamics.fixed_wing import FixedWing
from qp_control.utils import Utils

plt.style.use('seaborn-white')

sys.path.insert(1, os.path.abspath('.'))

which_data = input("Good data (1) or Last data (0): ")

n_state = 9
m_control = 4

fault_control_index = 1

dt = 0.01
n_sample = 10000
N1 = n_sample
N2 = 10000

nominal_params = config.FIXED_WING_PARAMS
fault = nominal_params["fault"]

state = torch.tensor([[100.0,
                       0.0,
                       0.0,
                       0.0,
                       0.0,
                       0.0,
                       0.0,
                       0.0,
                       0.0]])

goal = torch.tensor([[100.0,
                      0.0,
                      0.0,
                      0.0,
                      0.0,
                      0.0,
                      0.0,
                      0.0,
                      0.0]])

goal = np.array(goal).reshape(1, n_state)

dynamics = FixedWing(x=state, nominal_params=nominal_params, dt=dt, controller_dt=dt)
util = Utils(n_state=9, m_control=4, j_const=2, dyn=dynamics, dt=dt, params=nominal_params, fault=fault,
             fault_control_index=fault_control_index)

su, sl = dynamics.state_limits()

cbf = CBF(dynamics, n_state=n_state, m_control=m_control, fault=fault, fault_control_index=1)
nn_controller = NNController_new(n_state=9, m_control=4)
alpha = alpha_param(n_state=9)
if which_data == 0:
    if fault == 0:
        cbf.load_state_dict(torch.load('./data/FW_cbf_NN_weights.pth'))
        nn_controller.load_state_dict(torch.load('./data/FW_controller_NN_weights.pth'))
        alpha.load_state_dict(torch.load('./data/FW_alpha_NN_weights.pth'))
    else:
        cbf.load_state_dict(torch.load('./data/FW_cbf_FT_weights.pth'))
        nn_controller.load_state_dict(torch.load('./data/FW_controller_FT_weights.pth'))
        alpha.load_state_dict(torch.load('./data/FW_alpha_FT_weights.pth'))
else:
    try:
        if fault == 0:
            cbf.load_state_dict(torch.load('./good_data/data/FW_cbf_NN_weights.pth'))
            nn_controller.load_state_dict(torch.load('./good_data/data/FW_controller_NN_weights.pth'))
            alpha.load_state_dict(torch.load('./good_data/data/FW_alpha_NN_weights.pth'))
        else:
            cbf.load_state_dict(torch.load('./good_data/data/FW_cbf_FT_weights.pth'))
            nn_controller.load_state_dict(torch.load('./good_data/data/FW_controller_FT_weights.pth'))
            alpha.load_state_dict(torch.load('./good_data/data/FW_alpha_FT_weights.pth'))
    except:
        print("No good data available, evaluating on last data")
        if fault == 0:
            cbf.load_state_dict(torch.load('./data/FW_cbf_NN_weights.pth'))
            nn_controller.load_state_dict(torch.load('./data/FW_controller_NN_weights.pth'))
            alpha.load_state_dict(torch.load('./data/FW_alpha_NN_weights.pth'))
        else:
            cbf.load_state_dict(torch.load('./data/FW_cbf_FT_weights.pth'))
            nn_controller.load_state_dict(torch.load('./data/FW_controller_FT_weights.pth'))
            alpha.load_state_dict(torch.load('./data/FW_alpha_FT_weights.pth'))

cbf.eval()
nn_controller.eval()
alpha.eval()

safe_m, safe_l = dynamics.safe_limits(su, sl)

u_nominal = torch.zeros(N1 + N2, m_control)

deriv_safe = 0.0
safety_rate = 0.0
un_safety_rate = 0.0
correct_h_safe = 0.0
correct_h_un_safe = 0.0

iterations = 10

for k in range(iterations):
    # print(k)
    state_bndr = util.x_bndr(safe_m, safe_l, n_sample)

    state_bndr = state_bndr.reshape(N1, n_state)

    state = state_bndr + 2 * torch.randn(N1, n_state)

    for j in range(N2):
        state_temp = (su.clone() + sl.clone()) / 2 + 1 * torch.randn(1, n_state)
        state = torch.vstack((state, state_temp))

    state = state.reshape(N1 + N2, n_state)

    h, grad_h = cbf.V_with_jacobian(state)

    fx = dynamics._f(state, params=nominal_params)

    gx = dynamics._g(state, params=nominal_params)

    # u_n = util.nominal_controller(state=state, goal=goal, u_n=u_nominal, dyn=dynamics, constraints=constraints)

    # u_nominal = util.neural_controller(u_n, fx, gx, h, grad_h, fault_start=fault)

    # u_nominal = u_n.reshape(N1+N2, m_control)

    u = nn_controller(torch.tensor(state, dtype=torch.float32), torch.tensor(u_nominal, dtype=torch.float32))

    dsdt = fx + torch.matmul(gx, u.reshape(N1 + N2, m_control, 1))

    dsdt = torch.reshape(dsdt, (N1 + N2, n_state))

    alpha_p = alpha(state)

    dot_h = torch.matmul(grad_h.reshape(N1 + N2, 1, n_state),
                         dsdt.reshape(N1 + N2, n_state, 1))

    dot_h = dot_h.reshape(N1 + N2, 1)

    deriv_cond = dot_h + alpha_p * h

    deriv_safe += torch.sum((deriv_cond >= 0).reshape(N1+N2, 1) * util.is_safe(state).reshape(N1+N2, 1)) / (N1 + N2)

    safety_rate += torch.sum(util.is_safe(state)) / (N1 + N2)

    un_safety_rate += torch.sum(util.is_unsafe(state)) / (N1 + N2)

    correct_h_safe += torch.sum(util.is_safe(state).reshape(1, N1 + N2) * (h >= 0).reshape(1, N1 + N2)) / (N1 + N2)

    correct_h_un_safe += torch.sum(util.is_unsafe(state).reshape(1, N1 + N2) * (h < 0).reshape(1, N1 + N2)) / (N1 + N2)

print(safety_rate / iterations)

print(un_safety_rate / iterations)

print(correct_h_safe / safety_rate)

print(correct_h_un_safe / un_safety_rate)

print(deriv_safe / safety_rate)

# print(N_safe)
# import matplotlib.pyplot as plt
# alpha_index = 1 # import pdb; pdb.set_trace() state[:, 2:, :] = 0 plt.scatter(state[:, alpha_index,
# 0].detach().numpy(), state[:, 0, 0].detach().numpy(), c=util.is_safe(state).type(torch.float).squeeze().detach(
# ).numpy()) plt.show() plt.scatter(state[:, alpha_index, 0].detach().numpy(), state[:, 0, 0].detach().numpy(),
# c=h.squeeze().detach().numpy()) plt.colorbar()
# plt.show()
