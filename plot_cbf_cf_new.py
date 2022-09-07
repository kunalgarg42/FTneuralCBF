import os
import sys
sys.path.insert(1, os.path.abspath('.'))

import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import numpy as np
import torch
from qp_control.NNfuncgrad import CBF
from dynamics.Crazyflie import CrazyFlies


n_state = 12
m_control = 4

fault = int(input("Fault (1) or pre-fault (0):"))
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

state = torch.tensor([[10.0,
                    10.0,
                    10.0,
                    4.0,
                    4.0,
                    4.0,
                    np.pi / 2.0,
                    np.pi / 2.0,
                    np.pi / 2.0,
                    np.pi / 4.0,
                    np.pi / 4.0,
                    np.pi / 4.0]])

dynamics = CrazyFlies(x=state, nominal_params=nominal_params, dt=dt, controller_dt=dt)

if fault == 0:
    NN_cbf = CBF(dynamics, n_state=n_state, m_control=m_control)
    NN_cbf.load_state_dict(torch.load('./data/CF_cbf_NN_weights.pth'))
    NN_cbf.eval()
else:
    FT_cbf = CBF(dynamics, n_state=n_state, m_control=m_control)
    FT_cbf.load_state_dict(torch.load('./data/CF_cbf_FT_weights.pth'))
    FT_cbf.eval()

# create x, y, z state space
xl = -10
xu = 10
yl = -10
yu = 10
zl = 0
zu = 1
x_mesh = 50
y_mesh = 50
z_mesh = 50
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
if fault == 0:
    ax.set_title('min_{x,y} h(z) over z, no fault')
    plt.savefig('./plots/plotcbf_cf_NN.png')
else:
    ax.set_title('min_{x,y} h(z) over z, with fault')
    plt.savefig('./plots/plotcbf_cf_FT.png')
