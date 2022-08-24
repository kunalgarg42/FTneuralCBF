import numpy as np
import math
from CBF import CBF

from qp_control.setup_AB import setup_AB

from qpsolvers import solve_qp
from osqp import OSQP

from scipy.sparse import identity
from scipy.sparse import vstack, csr_matrix, csc_matrix

Q = csc_matrix(10*identity(5))
F = np.array([1]*5).reshape(5,1)

def QP_con(x,xr,fx,gx):

	A_init, B_init = setup_AB.AB_mat(x,xr,fx,gx)
	N = np.array(A_init).shape[0]
	m = np.array(gx).shape[1]
	A = []
	B = []

	act_cont = []
	for i in range(N):
		if abs(A_init[i][0])+abs(A_init[i][1])>0.0001:
			A.append(A_init[i])
			B.append(B_init[i])
	
	A = np.array(A)
	B = np.array(B)
	
	if B.shape[0]<1:
		u = solve_qp(Q, F, solver = "osqp")
	else:
		u = solve_qp(Q, F, A, B, solver="osqp")

	return [u[0], u[1]]