import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
from qp_control.NNfuncgrad import CBF
from dynamics.fixed_wing import FixedWing
from qp_control import config
plt.style.use('seaborn-white')
sys.path.insert(1, os.path.abspath('.'))

n_state = 9
m_control = 4

dt = 0.01
nominal_params = config.FIXED_WING_PARAMS

fault = nominal_params["fault"]

state = torch.tensor([[0.0,
                       0.0,
                       0.0,
                       0.0,
                       0.0,
                       0.0,
                       0.0,
                       0.0,
                       0.0]])

dynamics = FixedWing(x=state, nominal_params=nominal_params, dt=dt, controller_dt=dt)
if fault == 0:
    NN_cbf = CBF(dynamics, n_state=n_state, m_control=m_control)
    NN_cbf.load_state_dict(torch.load('./data/FW_cbf_NN_weights.pth'))
    NN_cbf.eval()
else:
    FT_cbf = CBF(dynamics, n_state=n_state, m_control=m_control)
    FT_cbf.load_state_dict(torch.load('./data/FW_cbf_FT_weights.pth'))
    FT_cbf.eval()

# create x, y, z state space
# FW: x = v, y = beta, z = alpha
xl = 0
xu = 200
yl = -np.pi / 6
yu = np.pi / 6
zl = -np.pi / 3
zu = np.pi / 3
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
h_store = torch.zeros((1, zlen))
state_new = state

hmin = 100
for j in range(0, xlen):
    for k in range(0, ylen):
        state_new[0, 0] = x[j]
        state_new[0, 2] = y[k]
        state = torch.vstack((state, state_new))
        # print(state)
bs = xlen * ylen + 1

for i in range(0, zlen):
    state[:, 1] = torch.ones(bs) * z[i]
    if fault == 0:
        h, _ = NN_cbf.V_with_jacobian(state.reshape(bs, n_state, 1))
    else:
        h, _ = FT_cbf.V_with_jacobian(state.reshape(bs, n_state, 1))
    hmin = torch.min(h)
    h_store[0, i] = hmin

# initialize fig
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(12, 8)

h_store = h_store.detach().numpy()
h_store = h_store.flatten()
print(z)
print(h_store)
ax.plot(z, h_store)
ax.set_xlabel('alpha (rad)')
ax.set_ylabel('min_{V,beta} h(alpha)')
if fault == 0:
    ax.set_title('min_{V,beta} h(alpha) over alpha, no fault')
    plt.savefig('./plots/plotcbf_fw_NN.png')
else:
    ax.set_title('min_{V,beta} h(alpha) over alpha, with fault')
    plt.savefig('./plots/plotcbf_fw_FT.png')
