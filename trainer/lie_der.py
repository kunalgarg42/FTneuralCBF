import numpy as np
import math
from train.CBF import B, V, LfB, LfV

n = 10


def Lie(xs, xr, r, fxs, gxs, stype):
    if stype == 'CBF' or stype == 'barrier' or stype == 'Barrier' or stype == 'B':
        h = B(xs, xr, r, n)
        Lf = LfB(xs, xr, r, fxs, n)
        Lg = LfB(xs, xr, r, gxs, n)
    # print(h)
    else:
        h = V(xs, xr, r)
        Lf = LfV(xs, xr, fxs)
        Lg = LfV(xs, xr, gxs)

    return h, Lf, Lg


class lie_der:
    pass
