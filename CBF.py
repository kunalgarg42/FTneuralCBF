import numpy as np
import torch


def B(state, obs, r):
	er = state - obs
	h = r * r - torch.linalg.norm(er) ** 2
	return h


def LfB(state, obs, f):
	N = f.shape[0]

	er = state - obs
	er = er.reshape(1, N)
	lie_V = torch.matmul(er, f)

	lie_B = -1 * lie_V

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


