import os
import sys
sys.path.insert(1, os.path.abspath('..'))
sys.path.insert(1, os.path.abspath('.'))
import matplotlib.pyplot as plt
import numpy as np
import torch
from dynamics.fixed_wing import FixedWing
from trainer import config
from trainer.constraints_fw import constraints
from trainer.NNfuncgrad import CBF, NNController_new, alpha_param
from trainer.utils import Utils


plt.style.use('seaborn-white')

which_data = int(input("Good data (1) or Last data (0): "))

n_state = 9
m_control = 4

fault_control_index = 1

dt = 0.01
n_sample = 50000
N1 = n_sample
N2 = 50000

nominal_params = config.FIXED_WING_PARAMS
fault = 1  # nominal_params["fault"]

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

cbf = CBF(dynamics, n_state=n_state, m_control=m_control, iter_NN=-1, fault=fault, fault_control_index=fault_control_index)
nn_controller = NNController_new(n_state=9, m_control=4)
alpha = alpha_param(n_state=9)
if which_data == 0:
    if fault == 0:
        cbf.load_state_dict(torch.load('./data/FW_cbf_NN_weightsCBF.pth'))
        # nn_controller.load_state_dict(torch.load('./data/FW_controller_NN_weightsCBF.pth'))
        # alpha.load_state_dict(torch.load('./data/FW_alpha_NN_weightsCBF.pth'))
    else:
        cbf.load_state_dict(torch.load('./data/FW_cbf_FT_weightsCBF.pth'))
        # nn_controller.load_state_dict(torch.load('./data/FW_controller_FT_weights.pth'))
        # alpha.load_state_dict(torch.load('./data/FW_alpha_FT_weights.pth'))
else:
    try:
        if fault == 0:
            cbf.load_state_dict(torch.load('./good_data/data/FW_cbf_NN_weightsCBF.pth'))
            # nn_controller.load_state_dict(torch.load('./good_data/data/FW_controller_NN_weights.pth'))
            # alpha.load_state_dict(torch.load('./good_data/data/FW_alpha_NN_weights.pth'))
        else:
            cbf.load_state_dict(torch.load('./good_data/data/FW_cbf_FT_weightsCBF.pth'))
            # nn_controller.load_state_dict(torch.load('./good_data/data/FW_controller_FT_weights.pth'))
            # alpha.load_state_dict(torch.load('./good_data/data/FW_alpha_FT_weights.pth'))
    except:
        print("No good data available, evaluating on last data")
        if fault == 0:
            cbf.load_state_dict(torch.load('./data/FW_cbf_NN_weightsCBF.pth'))
            # nn_controller.load_state_dict(torch.load('./data/FW_controller_NN_weights.pth'))
            # alpha.load_state_dict(torch.load('./data/FW_alpha_NN_weights.pth'))
        else:
            cbf.load_state_dict(torch.load('./data/FW_cbf_FT_weightsCBF.pth'))
            # nn_controller.load_state_dict(torch.load('./data/FW_controller_FT_weights.pth'))
            # alpha.load_state_dict(torch.load('./data/FW_alpha_FT_weights.pth'))

cbf.eval()
# nn_controller.eval()
# alpha.eval()

safe_m, safe_l = dynamics.safe_limits(su, sl)

u_nominal = torch.zeros(N1 + N2, m_control)

um, ul = dynamics.control_limits()
batch_size = N1 + N2
um = um.reshape(1, m_control).repeat(batch_size, 1)
ul = ul.reshape(1, m_control).repeat(batch_size, 1)

um = um.type(torch.FloatTensor)
ul = ul.type(torch.FloatTensor)

deriv_safe = 0.0
deriv_unsafe = 0.0
safety_rate = 0.0
un_safety_rate = 0.0
correct_h_safe = 0.0
correct_h_un_safe = 0.0

iterations = 10

for k in range(iterations):
    state_bndr = util.x_bndr(safe_m, safe_l, n_sample)

    state0 = state_bndr.reshape(N1, n_state)
    # state0 = state0 + 1 * torch.randn(N1, n_state)

    state1 = util.x_samples(su, sl, N2)
    state1 = state1.reshape(N2, n_state)  # + 5 * torch.randn(N2, n_state)

    state = torch.vstack((state0, state1))

    state = state.reshape(N1 + N2, n_state)

    h, grad_h = cbf.V_with_jacobian(state)

    h = h.reshape(N1 + N2, 1)

    fx = dynamics._f(state, params=nominal_params)

    gx = dynamics._g(state, params=nominal_params)

    # alpha_p = alpha.forward(state)
    # alpha_p = alpha_p.reshape(N1 + N2, 1)

    dot_h = util.doth_max_alpha(h, grad_h, fx, gx, um, ul)

    # dot_h = dot_h.reshape(N1 + N2, 1)

    deriv_cond = dot_h  # + alpha_p * h

    # print(deriv_cond.min())

    deriv_safe += torch.sum((deriv_cond >= 0).reshape(N1 + N2, 1) * util.is_safe(state).reshape(N1 + N2, 1)) / (N1 + N2)

    deriv_unsafe += torch.sum(
        (deriv_cond >= 0).reshape(N1 + N2, 1) * util.is_unsafe(state).reshape(N1 + N2, 1)) / (N1 + N2)

    safety_rate += torch.sum(util.is_safe(state)) / (N1 + N2)

    un_safety_rate += torch.sum(util.is_unsafe(state)) / (N1 + N2)

    correct_h_safe += torch.sum(util.is_safe(state).reshape(1, N1 + N2) * (h >= 0).reshape(1, N1 + N2)) / (N1 + N2)

    correct_h_un_safe += torch.sum(util.is_unsafe(state).reshape(1, N1 + N2) * (h < 0).reshape(1, N1 + N2)) / (N1 + N2)

print(safety_rate / iterations)

print(un_safety_rate / iterations)

print(correct_h_safe / safety_rate)

print(correct_h_un_safe / un_safety_rate)

print(deriv_safe / safety_rate)

print(deriv_unsafe / un_safety_rate)

# print(N_safe)
# import matplotlib.pyplot as plt
# alpha_index = 1 # import pdb; pdb.set_trace() state[:, 2:, :] = 0 plt.scatter(state[:, alpha_index,
# 0].detach().numpy(), state[:, 0, 0].detach().numpy(), c=util.is_safe(state).type(torch.float).squeeze().detach(
# ).numpy()) plt.show() plt.scatter(state[:, alpha_index, 0].detach().numpy(), state[:, 0, 0].detach().numpy(),
# c=h.squeeze().detach().numpy()) plt.colorbar()
# plt.show()
