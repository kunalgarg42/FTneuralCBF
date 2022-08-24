import numpy as np
import math
from CBF import CBF

from qp_control.setup_AB import setup_AB

from qpsolvers import solve_qp
from osqp import OSQP

from scipy.sparse import identity
from scipy.sparse import vstack, csr_matrix, csc_matrix


def QP_con(x,alphar,Ftr,Vr,gr,fx,gx):
	Q = csc_matrix(100*identity(5))
	F = np.array([0]*5).reshape(5,1)
	# act_cont = 0

	# A1, A2, A3, B1, B2, B3 = setup_AB.AB_mat(x,alphar, Ftr,Vr,gr,fx,gx)
	A_init, B_init = setup_AB.AB_mat(x,alphar, Ftr,Vr,gr,fx,gx)
	N = np.array(A_init).shape[0]
	m = np.array(gx).shape[1]

	# A = np.array([], dtype=float)
	A = []
	B = []

	# for i in range(N):
	# 	if abs(A_init[i][0])+abs(A_init[i][1])>0.01:
	# 		act_cont = act_cont + 2*i + 1
	act_cont = []
	for i in range(N):
		if abs(A_init[i][0])+abs(A_init[i][1])>0.01:
			# act_cont = act_cont.append([i])
			A.append(A_init[i])
			B.append(B_init[i])
	A = np.array(A)
	B = np.array(B)
	# A = np.array(A_init[act_cont])
	# B = np.array(B_init[act_cont])
	# if act_cont == 1:
	# 	A = A_init[0]
	# 	B = B_init[0]
	# elif act_cont == 3:
	# 	A = A_init[1]
	# 	B = B_init[1]
	# elif act_cont == 4:
	# 	A = np.array([A_init[0], A_init[1]])
	# 	B = np.array([B_init[0], B_init[1]])
	# elif act_cont == 5:
	# 	A = A_init[2]
	# 	B = B_init[2]
	# elif act_cont == 6:
	# 	A = np.array([A_init[0], A_init[2]])
	# 	B = np.array([B_init[0], B_init[2]])
	# elif act_cont == 8:
	# 	A = np.array([A_init[1], A_init[2]])
	# 	B = np.array([B_init[1], B_init[2]])
	# elif act_cont == 9:
	# 	A = np.array(A_init)
	# 	B = np.array(B_init)
	# else:
	# 	A = []
	# 	B = []

	u = solve_qp(Q, F, A, B, solver="osqp")

	return [u[0], u[1]]