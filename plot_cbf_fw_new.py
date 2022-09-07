import os
import sys
sys.path.insert(1, os.path.abspath('.'))

import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import numpy as np
import torch
from qp_control.NNfuncgrad_CF import CBF
from dynamics.fixed_wing import FixedWing


n_state = 9
m_control = 4
fault = int(input("Fault (1) or pre-fault(0):"))
dt = 0.01

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
h_store = torch.zeros((1,zlen))
state_new  = state

hmin = 100
for j in range(0,xlen):
    for k in range(0, ylen):
        state_new[0,0] = x[j]
        state_new[0,2] = y[k]
        state = torch.vstack((state,state_new))
            # print(state)
bs = xlen*ylen + 1

for i in range(0,zlen):
    state[:,1] = torch.ones(bs) * z[i]
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
ax.set_xlabel('alpha (rad)')
ax.set_ylabel('min_{V,beta} h(alpha)')
if fault == 0:
    ax.set_title('min_{V,beta} h(alpha) over alpha, no fault')
    plt.savefig('./plots/plotcbf_fw_NN.png')
else:
    ax.set_title('min_{V,beta} h(alpha) over alpha, with fault')
    plt.savefig('./plots/plotcbf_fw_FT.png')
