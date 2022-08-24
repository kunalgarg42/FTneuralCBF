import numpy as np
import math
import torch
from CBF import CBF
from qp_control.lie_der import lie_der

class constraints():

	def LfLg(x,xr,fx,gx):
		
		alphar = xr[0]
		Ftr = xr[1]
		Vr = xr[2]
		gr = xr[3]

		alpha = x[2]-x[1]

		v = x[0]
		gamma = x[1]
		theta = x[2]
		q = x[3]
		
		N = 4
		m = np.array(gx).shape[1]

		thetar = gr + alphar;

		xs = np.array(x).reshape(1,N) - np.array([0, 0, gamma, 0], dtype= float).reshape(1,N)
		xr = np.array([v, gamma, alphar, q], dtype = float).reshape(1,N)
		fxs = fx + np.array([0, 0, -fx[1], 0], dtype = float).reshape(4,1)
		gxs = np.array(gx).reshape(4,m)
		gxs[2] = gxs[2] - gx[1]

		h, Lfh, Lgh = lie_der.Lie(xs, xr, 0.5, fxs, gxs, 'CBF')
		

		xr = np.array([Vr, gamma, theta, q],dtype = float)
		V1, LfV1, LgV1 = lie_der.Lie(x, xr, 0.1, fx, gx, 'CLF')

		xr = np.array([v, gamma, thetar, q], dtype = float)
		V21, LfV21, LgV21 = lie_der.Lie(x, xr, 0.1, fx, gx, 'CLF')

		xs = np.array(x).reshape(1,N) + np.array([0, 0, q, 0], dtype= float).reshape(1,N)
		xr = np.array([v, gamma, thetar, q], dtype = float).reshape(1,N)
		fxs = np.array(fx, dtype = float).reshape(4,1) + np.array([0, 0, fx[3], 0], dtype = float).reshape(4,1)
		
		gxs = np.array(gx).reshape(4,m)
		gxs[3] = gxs[3] + gx[3]

		V22, LfV22, LgV22 = lie_der.Lie(xs, xr, 0.1, fxs, gxs, 'CLF')

		V2 = V21 + V22

		LfV2 = LfV21 + LfV22
		LgV2 = LgV21 + LgV22
		
		Lg = [[Lgh[0], Lgh[1]], [LgV1[0], LgV1[1]], [LgV2[0], LgV2[1]]] 

		if V1<0:
			V1 = 0

		if V2<0:
			V2 = 0

		Lf = [-Lfh, -LfV1-5.0*V1**0.5-5.0*V1**2.0, -LfV2-5.0*V2**0.5-5.0*V2**2.0]
		
		funV = np.diag([-h, -V1, -V2])
		
		Lg = np.array(Lg,dtype = float).reshape(3,2)
		
		# return [[Lgh[0], Lgh[1], -h, 0, 0], [LgV1[0], LgV1[1], 0, -LfV1, 0], [LgV2[0], LgV2[1], 0, 0, -LfV2]], [-Lfh, -LfV1, -LfV2]
		return funV, Lg, Lf

	def LfLg_new(x,xr,fx,gx,n_state,m_control,j_const,batch_size):
		x = x.detach().cpu().numpy()
		# xr = xr.detach().cpu().numpy()
		fx = fx.detach().cpu().numpy()
		gx = gx.detach().cpu().numpy()
		
		# alphar = xr[:, 0]
		# Ftr = xr[1]
		# Vr = xr[2]
		# gr = xr[3]

		# alpha = x[2]-x[1]

		# v = x[0]
		# gamma = x[1]
		# theta = x[2]
		# q = x[3]
		
		# N = 4
		# m = np.array(gx).shape[1]

		# thetar = gr + alphar;

		# xs = np.array(x).reshape(1,N) - np.array([0, 0, gamma, 0], dtype= float).reshape(1,N)
		# xr = np.array([v, gamma, alphar, q], dtype = float).reshape(1,N)
		# xs = x
		fxs = fx
		gxs = gx
		# gxs[2] = gxs[2] - gx[1]

		V, LfV, LgV = lie_der.Lie(x, xr, 0.5, fxs, gxs, 'CLF')

		# xr = xs
		# xr[:, 1] = 
		h, Lfh, Lgh = lie_der.Lie(x, np.array([0.3]*batch_size).reshape(batch_size,1), 0.5, fxs, gxs, 'CBF')
		

		# xr = np.array([Vr, gamma, theta, q],dtype = float)
		# V1, LfV1, LgV1 = lie_der.Lie(x, xr, 0.1, fx, gx, 'CLF')

		# xr = np.array([v, gamma, thetar, q], dtype = float)
		# V21, LfV21, LgV21 = lie_der.Lie(x, xr, 0.1, fx, gx, 'CLF')

		# xs = np.array(x).reshape(1,N) + np.array([0, 0, q, 0], dtype= float).reshape(1,N)
		# xr = np.array([v, gamma, thetar, q], dtype = float).reshape(1,N)
		# fxs = np.array(fx, dtype = float).reshape(4,1) + np.array([0, 0, fx[3], 0], dtype = float).reshape(4,1)
		
		# gxs = np.array(gx).reshape(4,m)
		# gxs[3] = gxs[3] + gx[3]

		# V22, LfV22, LgV22 = lie_der.Lie(xs, xr, 0.1, fxs, gxs, 'CLF')

		# V2 = V21 + V22

		# LfV2 = LfV21 + LfV22
		# LgV2 = LgV21 + LgV22
		# print(Lgh.shape)
		
		Lg = torch.vstack((torch.tensor(Lgh), torch.tensor(LgV))) 

		# print(V)

		if V<0:
			V = 0

		# if V2<0:
			# V2 = 0

		Lf = [-Lfh, -LfV-5.0*V**0.5-5.0*V**2.0]
		funv = torch.tensor([-h, -V])
		
		funV = torch.diag(funv)
		
		# Lg = np.array(Lg,dtype = float).reshape(3,2)
		
		# return [[Lgh[0], Lgh[1], -h, 0, 0], [LgV1[0], LgV1[1], 0, -LfV1, 0], [LgV2[0], LgV2[1], 0, 0, -LfV2]], [-Lfh, -LfV1, -LfV2]
		return funV, Lg, Lf
