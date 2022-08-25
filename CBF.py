import numpy as np
import torch


class CBF():

	def B(state,obs,r):
		N = state.shape[1]
		er = state-obs
		h = r * r - np.linalg.norm(er) ** 2
		return h

	def LfB(state,obs,f):
		N = f.shape[0]

		er = state - obs
		er = er.reshape(1,N)
		lie_V = np.matmul(er, f)
		
		lie_B = -1*lie_V

		return lie_B

	def V(state,ref,r):
		N = state.shape[1]
		er = state -ref
		V = np.linalg.norm(er) ** 2 -r ** 2
		return V

	def LfV(state,ref,f):
		N = f.shape[0]

		er = state - ref
		er = er.reshape(1,N)
		lie_V = np.matmul(er,f)
		return lie_V

