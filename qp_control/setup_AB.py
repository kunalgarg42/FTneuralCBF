import numpy as np
import math
from CBF import CBF
# from qp_control.lie_der import lie_der
from qp_control.constraints_fw import constraints
from scipy.sparse import identity
from scipy.sparse import vstack, csr_matrix, csc_matrix

class setup_AB(): 

	def AB_mat(x,xr,fx,gx):
	
		
		V, Lg, Lf = constraints.LfLg(x,xr,fx,gx)
		A = np.hstack((Lg, V))
		A = np.array(A,dtype = float)
		B = Lf

		return A, B