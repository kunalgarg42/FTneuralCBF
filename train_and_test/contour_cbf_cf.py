import os
import sys

sys.path.insert(1, os.path.abspath('.'))

import matplotlib.pyplot as plt

plt.style.use('seaborn-white')
import numpy as np
import torch
from trainer.NNfuncgrad_CF import CBF
from dynamics.Crazyflie import CrazyFlies

n_state = 12
m_control = 4

fault = int(input("Fault (1) or pre-fault (0):"))
fault_control_index = 1
dt = 0.01

nominal_params = {
    "m": 0.0299,
    "Ixx": 1.395 * 10 ** (-5),
    "Iyy": 1.395 * 10 ** (-5),
    "Izz": 2.173 * 10 ** (-5),
    "CT": 3.1582 * 10 ** (-10),
    "CD": 7.9379 * 10 ** (-12),
    "d": 0.03973,
    "fault": fault, }

state = torch.tensor([[2.0,
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

dynamics = CrazyFlies(x=state, nominal_params=nominal_params, dt=dt, controller_dt=dt)

if fault == 0:
    NN_cbf = CBF(dynamics, n_state=n_state, m_control=m_control, fault=fault, fault_control_index=fault_control_index)
    NN_cbf.load_state_dict(torch.load('./data/CF_cbf_NN_weights.pth'))
    NN_cbf.eval()
else:
    FT_cbf = CBF(dynamics, n_state=n_state, m_control=m_control, fault=fault, fault_control_index=fault_control_index)
    FT_cbf.load_state_dict(torch.load('./data/CF_cbf_FT_weights.pth'))
    FT_cbf.eval()

# h contour in 2D: z and w
zl = -1
zu = 10
# wl = -np.pi / 2
# wu = np.pi / 2
# w_ind = 4
wl = -5
wu = 5
w_ind = 8
z_mesh = 20
w_mesh = 20
z = np.linspace(zl, zu, z_mesh)
w = np.linspace(wl, wu, w_mesh)

# print(w)
# Get value of h
zlen = z.size
wlen = w.size
h_store = np.zeros((zlen + 1, wlen))
state_new = state

for j in range(0, zlen):
    state_new[0, 2] = z[j]
    state = torch.vstack((state, state_new))
bs = zlen + 1

state_all = state.clone()
for i in range(0, wlen):
    state_all[:, w_ind] = torch.ones(bs) * w[i]
    if fault == 0:
        # print(state_all)
        h, _ = NN_cbf.V_with_jacobian(state.reshape(bs, n_state, 1))
    else:
        h, _ = FT_cbf.V_with_jacobian(state.reshape(bs, n_state, 1))

    h = h.detach().numpy()
    h = h.flatten()
    h_store[:, i] = np.copy(h)

# print(h_store)

# initialize fig
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(12, 8)

X, Y = np.meshgrid(z, w)
Z = np.copy(h_store[1::])

fig, ax = plt.subplots()
CS = ax.contour(X, Y, np.transpose(Z))
ax.clabel(CS, inline=True, fontsize=10)

if fault == 0:
    ax.set_title('CBF Contour over z and w, no fault')
    plt.savefig('./plots/Con_cbf_cf_NN.png')
else:
    ax.set_title('CBF Contour over z and w, with fault')
    plt.savefig('./plots/Con_cbf_cf_FT.png')
