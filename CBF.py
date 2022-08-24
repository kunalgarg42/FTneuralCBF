import numpy as np
import torch


class CBF():
	# def _init_(state):
	# 	super().__init__()

	def B(state,obs,r):
		N = state.shape[1]
		# print(state.shape)
		# print((state-obs))
		er = state-obs
		# print(er)
		h = r * r - np.linalg.norm(er) ** 2
		# print(h)
		return h

	def LfB(state,obs,f):
		# print(f.shape)
		N = f.shape[0]
		m = f.shape[1]

		# print(f)

		# print(N)
		# print(m)
		er = state - obs
		er = er.reshape(1,N)
		# lie_V = er * f
		lie_V = np.matmul(er, f)
		
		# er = np.array(state).reshape(1,N)-np.array(obs).reshape(1,N)
		# er = np.array(er).reshape(1,N)
		# f = np.array(f).reshape(N,m)
		# lie_V = np.dot(er,f)
		lie_B = -1*lie_V

		# print(lie_B.shape)
		# lie_B = -2*(state-obs)*f
		return lie_B

	def V(state,ref,r):
		# er = state - ref
		N = state.shape[1]
		er = state -ref
		V = np.linalg.norm(er) ** 2 -r ** 2
		# print(V)
		return V

	def LfV(state,ref,f):
		# print(np.array(f).shape)
		N = f.shape[1]
		m = f.shape[0]

		er = state - obs
		er = er.reshape(N,1)
		lie_V = np.matmul(er,f)
		return lie_V

