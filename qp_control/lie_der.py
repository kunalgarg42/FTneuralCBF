import numpy as np
import math
from CBF import B, V, LfB, LfV


def Lie(xs, xr, r, fxs, gxs, stype):
    if stype == 'CBF' or stype == 'barrier' or stype == 'Barrier' or stype == 'B':
        h = B(xs, xr, r)
        Lf = LfB(xs, xr, fxs)
        Lg = LfB(xs, xr, gxs)
    # print(h)
    elif stype == 'CLF' or stype == 'lyapunov' or stype == 'Lyapunov' or stype == 'V':
        h = V(xs, xr, r)

        Lf = LfV(xs, xr, fxs)
        Lg = LfV(xs, xr, gxs)

    # print(h)
    return h, Lf, Lg


class lie_der:
    pass
