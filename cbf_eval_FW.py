import os
import sys
sys.path.insert(1, os.path.abspath('.'))

import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import numpy as np
import torch
from qp_control.NNfuncgrad import CBF
from dynamics.fixed_wing import FixedWing
from qp_control.utils import Utils


n_state = 9
m_control = 4

fault = int(input("Fault (1) or pre-fault (0):"))

fault_control_index = 1

dt = 0.01
N = 100000

nominal_params = {
    "m": 1000.0,
    "g": 9.8,
    "Ixx": 100,
    "Iyy": 100,
    "Izz": 1000,
    "Ixz": 0.1,
    "S": 25,
    "b": 4,
    "bar_c": 4,
    "rho": 1.3,
    "Cd0": 0.0434,
    "Cda": 0.22,
    "Clb": -0.13,
    "Clp": -0.505,
    "Clr": 0.252,
    "Clda": 0.0855,
    "Cldr": -0.0024,
    "Cm0": 0.135,
    "Cma": -1.50,
    "Cmq": -38.2,
    "Cmde": -0.992,
    "Cnb": 0.0726,
    "Cnp": -0.069,
    "Cnr": -0.0946,
    "Cnda": 0.7,
    "Cndr": -0.0693,
    "Cyb": -0.83,
    "Cyp": 1,
    "Cyr": 1,
    "Cydr": 1,
    "Cz0": 0.23,
    "Cza": 4.58,
    "Czq": 1,
    "Czde": 1,
    "Cx0": 1,
    "Cxq": 1,
    "fault": fault,}

state = torch.tensor([[100.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0]])


dynamics = FixedWing(x=state, nominal_params=nominal_params, dt=dt, controller_dt=dt)
util = Utils(n_state=9, m_control=4, j_const=2, dyn=dynamics, dt=dt, params=nominal_params, fault=fault,
                 fault_control_index=fault_control_index)

su, sl = dynamics.state_limits()

if fault == 0:
    cbf = CBF(dynamics, n_state=n_state, m_control=m_control,fault = 0, fault_control_index = 1)
    cbf.load_state_dict(torch.load('./data/FW_cbf_NN_weights.pth'))
    cbf.eval()
else:
    cbf = CBF(dynamics, n_state=n_state, m_control=m_control, fault =0, fault_control_index = 1)
    cbf.load_state_dict(torch.load('./data/FW_cbf_FT_weights.pth'))
    cbf.eval()

safety_rate = 0.0
correct_h = 0.0
# for i in range(N):
#     state = (su.clone().reshape(1,n_state) + sl.clone().reshape(1,n_state)) / 2 + torch.rand(1,n_state)
#     h, _ = cbf.V_with_jacobian(state)
#     safety_rate = safety_rate * (1-1 / N) + int(util.is_safe(state)) / N
#     cor_h = int(util.is_safe(state) * h >= 0)
#     correct_h = correct_h * (1 - 1 / N) +  cor_h / N

state = torch.tensor([]).reshape(0,n_state) #torch.zeros(N,n_state).reshape(N,n_state)
for j in range(N):
    state_N = (su.clone() + sl.clone()) / 2 + 1 * torch.randn(1, n_state)
    state = torch.vstack((state,state_N))

state = state.reshape(N,n_state,1)

h, _  = cbf.V_with_jacobian(state)

safety_rate = torch.sum(util.is_safe(state)) / N

un_safety_rate = torch.sum(torch.logical_not(util.is_safe(state))) / N

correct_h_safe = torch.sum(util.is_safe(state).reshape(1, N) * (h >= 0).reshape(1, N)) / N
correct_h_un_safe = torch.sum(util.is_unsafe(state).reshape(1, N) * (h < 0).reshape(1, N)) / N

print(safety_rate)
print(un_safety_rate)
print(correct_h_safe / safety_rate)
print(correct_h_un_safe / un_safety_rate)

# import matplotlib.pyplot as plt

# alpha_index = 1
# # import pdb; pdb.set_trace()
# state[:, 2:, :] = 0
# plt.scatter(state[:, alpha_index, 0].detach().numpy(), state[:, 0, 0].detach().numpy(), c=util.is_safe(state).type(torch.float).squeeze().detach().numpy())
# plt.show()
# plt.scatter(state[:, alpha_index, 0].detach().numpy(), state[:, 0, 0].detach().numpy(), c=h.squeeze().detach().numpy())
# plt.colorbar()

# plt.show()
