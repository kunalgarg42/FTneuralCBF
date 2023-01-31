import os
import sys

sys.path.insert(1, os.path.abspath('.'))
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

plt.style.use('seaborn-white')
import numpy as np
import torch
from trainer.NNfuncgrad_CF import CBF
from dynamics.Crazyflie import CrazyFlies
import seaborn as sns
from trainer import config
from trainer.utils import Utils

n_state = 12
m_control = 4
fault = 0
fault_control_index = 1
dt = 0.001

nominal_params = config.CRAZYFLIE_PARAMS

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

dynamics = CrazyFlies(x=state, nominal_params=nominal_params, dt=dt, goal=state)
util = Utils(
        n_state=n_state,
        m_control=m_control,
        dyn=dynamics,
        params=nominal_params,
        fault=fault,
        fault_control_index=fault_control_index,
        j_const=2,
    )

# if fault == 0:
NN_cbf = CBF(dynamics, n_state=n_state, m_control=m_control, fault=0, fault_control_index=fault_control_index)
NN_cbf.load_state_dict(torch.load('./good_data/data/CF_cbf_NN_weightsCBF_FxTS.pth'))
NN_cbf.eval()
# else:
FT_cbf = CBF(dynamics, n_state=n_state, m_control=m_control, fault=1, fault_control_index=fault_control_index)
FT_cbf.load_state_dict(torch.load('./good_data/data/CF_cbf_FT_weightsCBF.pth'))
FT_cbf.eval()

sm, sl = dynamics.state_limits()


# h contour in 2D: z and w
zl = -1
zu = 11
z_mesh = 200
z = np.linspace(zl, zu, z_mesh)

# print(w)
# Get value of h
zlen = z.size
h_pre_store = np.zeros((zlen + 1, 1))
h_post_store = np.zeros((zlen + 1, 1))
state_new = state.clone()

bs = zlen + 1

for j in range(0, zlen):
    # state_new = util.x_samples(sm, sl, 1)
    state_new[0, 2] = z[j].copy()

    # state_new[0, 8] = w[j]
    state = torch.vstack((state, state_new))

# state_all = state.clone()
# for i in range(0, wlen):
#     state_all[:, w_ind] = torch.ones(bs) * w[i].copy()
    # if fault == 0:
        # print(state_all)
h_pre, _ = NN_cbf.V_with_jacobian(state.reshape(bs, n_state, 1))
# else:
h_post, _ = FT_cbf.V_with_jacobian(state.reshape(bs, n_state, 1))

h_pre = h_pre.detach().numpy()
h_pre = h_pre.flatten()
h_pre_store = np.copy(h_pre)
h_post = h_post.detach().numpy()
h_post = h_post.flatten()
h_post_store = np.copy(h_post)

# print(h_store)

len_h = len(h_pre_store)
h_pre_store = h_pre_store.reshape(len_h, 1)
h_post_store = h_post_store.reshape(len_h, 1)

z = np.arange(-1, 11, 11 / len_h)
z = z[0:len_h]

plt.rcParams.update({'font.size': 22})

# initialize fig
fig = plt.figure(figsize=(20, 14))
fig.tight_layout(pad=1.15)
z_ax = fig.subplots(1, 1)
colors = sns.color_palette()

# Plot the altitude and CBF value on one axis

z_ax.plot(z, h_pre_store, linewidth=4.0, label="$h_{pre}$", color=colors[0])
z_ax.plot(z, h_post_store, linewidth=4.0, label="$h_{post}$", color=colors[1])
z_ax.tick_params(axis="y", labelcolor=colors[0], labelsize = 25)
z_ax.tick_params(axis="x", labelsize = 25)


nofault_sm, nofault_sl = dynamics.safe_limits(sm, sl, fault=0)

fault_sm, fault_sl = dynamics.safe_limits(sm ,sl, fault=1)

z_ax.fill_between(
        [nofault_sl[2], nofault_sm[2]],
        [-2.0, -2.0],
        [2.0, 2.0],
        color="grey",
        alpha=0.5,
        label="Pre-Fault $X_{safe}$",
    )

z_ax.fill_between(
        [fault_sl[2], fault_sm[2]],
        [-2.0, -2.0],
        [2.0, 2.0],
        color="grey",
        alpha=0.2,
        label="Post-Fault $X_{safe}$",
    )

z_ax.fill_between(
        [0, 0.2],
        [-2.0, -2.0],
        [2.0, 2.0],
        color="blue",
        alpha=0.5,
        label="$X_{unsafe}$",
    )

# z_ax.fill_between(
#         [9.5, 11],
#         [-2.0, -2.0],
#         [2.0, 2.0],
#         color="blue",
#         alpha=0.5,
#     )

z_ax.plot(
        z,
        0 * z,
        color="k",
        linestyle="--",
        linewidth=4.0,
        label="h = 0",
    )

z_ax.legend(ncol = 3, fontsize = 30)
z_ax.set_xlim(0, 20)

# if fault == 0:
z_ax.set_title('CBF plots over z', fontsize = 45)

plt.xlabel('CBF h(x)', fontsize = 40)
plt.ylabel('Altitude z', fontsize = 40)

plt.savefig('./plots/Cbf_cf_both.png')
