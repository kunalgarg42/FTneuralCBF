import os
import sys
# import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(1, os.path.abspath('..'))
sys.path.insert(1, os.path.abspath('.'))

from trainer.NNfuncgrad_CF import CBF, NNController_new, alpha_param
from dynamics.Crazyflie import CrazyFlies
from trainer.utils import Utils
from trainer import config

# plt.style.use('seaborn-white')

which_data = int(input("Good data (1) or Last data (0): "))

n_state = 12
m_control = 4

dt = 0.01
n_sample = 10000
N1 = n_sample
N2 = 10000

nominal_params = config.CRAZYFLIE_PARAMS

fault = 1  # nominal_params["fault"]

fault_control_index = 1

state0 = torch.tensor([[2.0,
                        2.0,
                        3.5,
                        0.0,
                        0.0,
                        0.0,
                        np.pi / 20.0,
                        np.pi / 20.0,
                        np.pi / 20.0,
                        np.pi / 40.0,
                        np.pi / 40.0,
                        np.pi / 40.0]])

dynamics = CrazyFlies(x=state0, nominal_params=nominal_params, goal=state0)
util = Utils(n_state=12, m_control=4, j_const=2, dyn=dynamics, params=nominal_params, fault=fault,
             fault_control_index=fault_control_index)

su, sl = dynamics.state_limits()
cbf = CBF(dynamics, n_state=n_state, m_control=m_control, fault=fault, fault_control_index=0)
# nn_controller = NNController_new(n_state=n_state, m_control=m_control)
# alpha = alpha_param(n_state=n_state)

if which_data == 0:
    if fault == 0:
        cbf.load_state_dict(torch.load('./data/CF_cbf_NN_weightsCBF.pth'))
        # nn_controller.load_state_dict(torch.load('./data/CF_controller_NN_weights.pth'))
        # alpha.load_state_dict(torch.load('./data/CF_alpha_NN_weights.pth'))
    else:
        cbf.load_state_dict(torch.load('./data/CF_cbf_FT_weightsCBF.pth'))
        # nn_controller.load_state_dict(torch.load('./data/CF_controller_FT_weights.pth'))
        # alpha.load_state_dict(torch.load('./data/CF_alpha_FT_weights.pth'))
else:
    try:
        if fault == 0:
            cbf.load_state_dict(torch.load('./good_data/data/CF_cbf_NN_weightsCBF.pth'))
            # nn_controller.load_state_dict(torch.load('./good_data/data/CF_controller_NN_weights.pth'))
            # alpha.load_state_dict(torch.load('./good_data/data/CF_alpha_NN_weights.pth'))
        else:
            cbf.load_state_dict(torch.load('./good_data/data/CF_cbf_FT_weightsCBF.pth'))
            # nn_controller.load_state_dict(torch.load('./good_data/data/CF_controller_FT_weights.pth'))
            # alpha.load_state_dict(torch.load('./good_data/data/CF_alpha_FT_weights.pth'))
    except:
        print("No good data available, evaluating on last data")
        if fault == 0:
            cbf.load_state_dict(torch.load('./data/CF_cbf_NN_weightsCBF.pth'))
            # nn_controller.load_state_dict(torch.load('./data/CF_controller_NN_weights.pth'))
            # alpha.load_state_dict(torch.load('./data/CF_alpha_NN_weights.pth'))
        else:
            cbf.load_state_dict(torch.load('./data/CF_cbf_FT_weightsCBF.pth'))
            # nn_controller.load_state_dict(torch.load('./data/CF_controller_FT_weights.pth'))
            # alpha.load_state_dict(torch.load('./data/CF_alpha_FT_weights.pth'))

cbf.eval()
# nn_controller.eval()
# alpha.eval()


safe_m, safe_l = dynamics.safe_limits(su, sl)

um, ul = dynamics.control_limits()
batch_size = N1 + N2
um = um.reshape(1, m_control).repeat(batch_size, 1)
ul = ul.reshape(1, m_control).repeat(batch_size, 1)

um = um.type(torch.FloatTensor)
ul = ul.type(torch.FloatTensor)

# u_nominal = torch.zeros(N1 + N2, m_control)

deriv_safe = 0.0
safety_rate = 0.0
un_safety_rate = 0.0
correct_h_safe = 0.0
correct_h_un_safe = 0.0

iterations = 10

for k in range(iterations):
    # print(k)
    state_bndr = util.x_samples(su, sl, n_sample)
    # state_bndr = dynamics.sample_state_space(n_sample)
    state_bndr = state_bndr.reshape(N1, n_state)

    state = state_bndr + 1 * torch.randn(N1, n_state)

    for j in range(N2):
        state_temp = (su.clone() + sl.clone()) / 2 + 5 * torch.randn(1, n_state)
        state = torch.vstack((state, state_temp))

    state = state.reshape(N1 + N2, n_state)

    h, grad_h = cbf.V_with_jacobian(state)

    h = h.reshape(N1 + N2, 1)

    fx = dynamics._f(state, params=nominal_params)
    gx = dynamics._g(state, params=nominal_params)

    # alpha_p = alpha.forward(state)
    # alpha_p = alpha_p.reshape(N1 + N2, 1)

    # dot_h = torch.matmul(grad_h.reshape(N1 + N2, 1, n_state),
    #                      dsdt.reshape(N1 + N2, n_state, 1))

    # dot_h = util.doth_max(grad_h, fx, gx, um, ul)
    # dot_h = dot_h.reshape(N1 + N2, 1)

    # deriv_cond = dot_h + alpha_p * h
    dot_h = util.doth_max_alpha(h, grad_h, fx, gx, um, ul)

    deriv_cond = dot_h

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
