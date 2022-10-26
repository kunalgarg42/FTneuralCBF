import numpy as np
import math
import torch
# from CBF import CBF
from qp_control.lie_der import Lie


def LfLg_new(x, xr, fx, gx, batch_size, alpha):
    # x = x.detach().cpu().numpy()
    # fx = fx.detach().cpu().numpy()
    # gx = gx.detach().cpu().numpy()

    fxs = fx
    gxs = gx

    alpha_m = alpha[0]
    alpha_l = alpha[1]
    alpha_mid = (alpha_l + alpha_m) / 2
    alpha_range = alpha_m - alpha_l

    V, LfV, LgV = Lie(x, xr, 0.5, fxs, gxs, 'CLF')

    xobs = x.clone()
    xobs[:, 1] = torch.ones(batch_size, 1) * alpha_mid

    h1, Lfh1, Lgh1 = Lie(x, xobs, alpha_range, fxs, gxs, 'CBF')

    # xobs[:,1] = np.array([alpha_l]*batch_size).reshape(batch_size,1)
    # h2, Lfh2, Lgh2 = lie_der.Lie(x, xobs, 0.5, fxs, gxs, 'CBF')

    Lg = torch.vstack((torch.tensor(Lgh1), torch.tensor(LgV)))

    if V < 0:
        V = 0.0 * V.clone()

    Lf = torch.hstack((-Lfh1, -LfV - 5.0 * V ** 0.5 - 5.0 * V ** 2.0))
    if x.get_device() == 0:
        funv = torch.tensor([-h1, -V]).cuda()
    else:
        funv = torch.tensor([-h1, -V])

    funV = torch.diag(funv)

    return funV, Lg, Lf


class constraints:
    def __init__(self):
        super().__init__()
        self.some = 0
