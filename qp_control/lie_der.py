import numpy as np
import math
from CBF import CBF

class lie_der():
	def Lie(xs,xr,r,fxs,gxs,stype):
		if stype == 'CBF' or stype == 'barrier' or stype == 'Barrier' or stype == 'B':
			h = CBF.B(xs, xr, r)

			Lf = CBF.LfB(xs,xr,fxs)
			Lg = CBF.LfB(xs,xr,gxs)
			# print(h)
		elif stype == 'CLF' or stype == 'lyapunov' or stype == 'Lyapunov' or stype == 'V':
			h = CBF.V(xs, xr, r)

			Lf = CBF.LfV(xs,xr,fxs)
			Lg = CBF.LfV(xs,xr,gxs)

			# print(h)
		return h, Lf, Lg

