import os
import sys
sys.path.insert(1, os.path.abspath('.'))

import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import numpy as np
import torch
from qp_control.NNfuncgrad_CF import CBF
from dynamics.Crazyflie import CrazyFlies
from qp_control.utils_crazy import Utils


n_state = 12
m_control = 4

fault = int(input("Fault (1) or pre-fault (0):"))

fault_control_index = 1

dt = 0.01
N = 10000

nominal_params = {
    "m": 0.0299,
    "Ixx": 1.395 * 10**(-5),
    "Iyy": 1.395 * 10**(-5),
    "Izz": 2.173 * 10**(-5),
    "CT": 3.1582 * 10**(-10),
    "CD": 7.9379 * 10**(-12),
    "d": 0.03973,
    "fault": fault,}

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

dynamics = CrazyFlies(x=state0, nominal_params=nominal_params, dt=dt, controller_dt=dt)
util = Utils(n_state=12, m_control=4, j_const=2, dyn=dynamics, dt=dt, params=nominal_params, fault=fault,
                 fault_control_index=fault_control_index)

su, sl = dynamics.state_limits()

if fault == 0:
    cbf = CBF(dynamics, n_state=n_state, m_control=m_control, fault = 0, fault_control_index = 1)
    cbf.load_state_dict(torch.load('./data/CF_cbf_NN_weights.pth'))
    cbf.eval()
else:
    cbf = CBF(dynamics, n_state=n_state, m_control=m_control, fault = 1, fault_control_index = 1)
    cbf.load_state_dict(torch.load('./data/CF_cbf_FT_weights.pth'))
    cbf.eval()

safety_rate = 0.0
correct_h = 0.0


state = torch.tensor([]).reshape(0,n_state) #torch.zeros(N,n_state).reshape(N,n_state)
for j in range(N):
    state_N = state0.reshape(1,n_state) + 1 * torch.randn(1, n_state)
    state = torch.vstack((state,state_N))

state = state.reshape(N,n_state,1)

h, _  = cbf.V_with_jacobian(state)

safety_rate = torch.sum(util.is_safe(state)) / N

un_safety_rate = torch.sum(torch.logical_not(util.is_safe(state))) / N

correct_h_safe = torch.sum(util.is_safe(state).reshape(1, N) * (h >= 0).reshape(1, N)) / N
correct_h_un_safe = torch.sum(util.is_unsafe(state).reshape(1, N) * (h < 0).reshape(1, N)) / N

print(safety_rate)
print(un_safety_rate)
print(correct_h_safe / (safety_rate + 1e-5))
print(correct_h_un_safe / (un_safety_rate + 1e-5))

