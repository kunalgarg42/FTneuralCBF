import torch
from trainer.lie_der import Lie


def LfLg_new(x, xr, fx, gx, sm, sl):
    # x = x.detach().cpu().numpy()
    # fx = fx.detach().cpu().numpy()
    # gx = gx.detach().cpu().numpy()

    fxs = fx
    gxs = gx

    # alpha_m = sm
    # alpha_l = alpha[1]
    safe_mid = (sm + sl) / 2
    safe_range = (sm - sl) / 2

    V, LfV, LgV = Lie(x, xr, 0.5, fxs, gxs, 'CLF')

    h1, Lfh1, Lgh1 = Lie(x, safe_mid, safe_range, fxs, gxs, 'CBF')

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