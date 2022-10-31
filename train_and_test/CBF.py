import numpy as np
import torch


def B(state, obs, r, n):
	# er = state - obs
	# h = r * r - torch.linalg.norm(er) ** 2
	assert int(n) == n
	er = (state - obs) / r
	bs = er.shape[0]
	n = er.shape[1]
	er = torch.matmul(er, er.reshape(n, bs))
	h = er ** n - 1
	return h


def LfB(state, obs, r, f, n):
	# N = f.shape[0]
	#
	# er = state - obs
	# er = er.reshape(1, N)
	# lie_V = torch.matmul(er, f)
	#
	# lie_B = -1 * lie_V
	er = (state - obs) / r
	bs = er.shape[0]
	N = er.shape[1]
	lie_B = n * torch.matmul(er, er.reshape(N, bs)) ** (n-1)
	lie_B = - lie_B * torch.matmul(er, f)
	return lie_B


def V(state, ref, r):
	er = state - ref
	v = torch.linalg.norm(er) ** 2 - r ** 2
	return v


def LfV(state, ref, f):
	N = f.shape[0]

	er = state - ref
	er = er.reshape(1, N)
	lie_V = torch.matmul(er, f)
	return lie_V

# class CBF:


