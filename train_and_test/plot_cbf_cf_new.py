import os
import sys
sys.path.insert(1, os.path.abspath('..'))

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
    "Ixx": 1.395 * 10**(-5),
    "Iyy": 1.395 * 10**(-5),
    "Izz": 2.173 * 10**(-5),
    "CT": 3.1582 * 10**(-10),
    "CD": 7.9379 * 10**(-12),
    "d": 0.03973,
    "fault": fault,}

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
    NN_cbf = CBF(dynamics, n_state=n_state, m_control=m_control,fault = fault, fault_control_index = fault_control_index)
    NN_cbf.load_state_dict(torch.load('./data/CF_cbf_NN_weights.pth'))
    NN_cbf.eval()
else:
    FT_cbf = CBF(dynamics, n_state=n_state, m_control=m_control, fault = fault, fault_control_index = fault_control_index)
    FT_cbf.load_state_dict(torch.load('./data/CF_cbf_FT_weights.pth'))
    FT_cbf.eval()

# create x, y, z state space
xl = -1
xu = 1
yl = -1
yu = 1
zl = -1
zu = 10
x_mesh = 50
y_mesh = 50
z_mesh = 100
x = np.linspace(xl, xu, x_mesh)
y = np.linspace(yl, yu, y_mesh)
z = np.linspace(zl, zu, z_mesh)

# Get min_{x,y} h(z) over a range of z
zlen = z.size
xlen = x.size
ylen = y.size
h_store = torch.zeros((1,zlen))
state_new  = state

hmin = 100
for j in range(0,xlen):
    for k in range(0, ylen):
        state_new[0,0] = x[j]
        state_new[0,1] = y[k]
        state = torch.vstack((state,state_new))
            # print(state)
bs = xlen*ylen + 1

for i in range(0,zlen):
    state[:,2] = torch.ones(bs) * z[i]
    if fault == 0:
        h, _ = NN_cbf.V_with_jacobian(state.reshape(bs, n_state, 1))
    else:
        h, _ = FT_cbf.V_with_jacobian(state.reshape(bs, n_state, 1))
    hmin = torch.min(h)
    #print(h)
    h_store[0,i] = hmin


# initialize fig
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(12, 8)

h_store = h_store.detach().numpy()
h_store = h_store.flatten()
print(z)
print(h_store)
ax.plot(z, h_store)
ax.set_xlabel('z (m)')
ax.set_ylabel('min_{x,y} h(z)')

h_B = np.copy(h_store)
ind_max = np.where(h_B == np.amax(h_B))
h_B = np.where(h_B > 0, h_B, 100)
# print(h_B)
ind_1 = np.where(h_B == np.amin(h_B))
print(z[ind_1])
h_B[ind_1] = 100
ind_2 = np.where(h_B == np.amin(h_B))
if z[ind_1] < z[ind_max]:
    while z[ind_2] < z[ind_max]:
        h_B[ind_2] = 100
        ind_2 = np.where(h_B == np.amin(h_B))
if z[ind_1] > z[ind_max]:
    while z[ind_2] > z[ind_max]:
        h_B[ind_2] = 100
        ind_2 = np.where(h_B == np.amin(h_B))

print(z[ind_2])
plt.axvline(x=z[ind_1], color='b',linewidth=5)
plt.axvline(x=z[ind_2], color='b',linewidth=5)


if fault == 0:
    safe_alpha = 8
    safe_alpha_l = 1.5
    unsafe_alpha = 10
    unsafe_alpha_l = 1.0
    plt.axvline(x=safe_alpha, color='g')
    plt.axvline(x=safe_alpha_l, color='g')
    plt.axvline(x=unsafe_alpha, color='r')
    plt.axvline(x=unsafe_alpha_l, color='r')
    ax.set_title('min_{x,y} h(z) over z, no fault')
    plt.savefig('./plots/plotcbf_cf_NN.png')
else:
    safe_alpha = 10
    safe_alpha_l = 1.0
    unsafe_alpha = 12
    unsafe_alpha_l = 0.5
    plt.axvline(x=safe_alpha, color='g')
    plt.axvline(x=safe_alpha_l, color='g')
    plt.axvline(x=unsafe_alpha, color='r')
    plt.axvline(x=unsafe_alpha_l, color='r')
    ax.set_title('min_{x,y} h(z) over z, with fault')
    plt.savefig('./plots/plotcbf_cf_FT.png')
