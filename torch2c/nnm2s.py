import os
import sys
sys.path.insert(1, os.path.abspath('.'))


import torch
import numpy as np
import random
from sympy.solvers import nsolve
from sympy import Symbol, sin, cos
import matplotlib.pyplot as plt
from dynamics.fixed_wing_dyn import fw_dyn_ext, fw_dyn
from dynamics.fixed_wing import FixedWing
from qp_control import config
from CBF import CBF
from qp_control.qp_con import QP_con
from qp_control.constraints_fw import constraints
import math

from qp_control.datagen import Dataset_with_Grad
from qp_control.utils import Utils

from qp_control.NNfuncgrad import CBF, alpha_param, NNController_new


# import cProfile
# cProfile.run('foo()')



xg = torch.tensor([[100.0, 
0.2, 
0.0, 
0.0, 
0.0, 
0.0,
0.0,
0.0,
0.0]])


x0 = torch.tensor([[50.0, 
0.1, 
0.1, 
0.4, 
0.5, 
0.2,
0.1,
0.5,
0.9]])


dt = 0.01
n_state = 9
m_control = 4
fault = 0

nominal_params = {
	"m": 1000.0, 
	"g": 9.8, 
	"Ixx": 100, 
	"Iyy": 100, 
	"Izz": 1000, 
	"Ixz": 0.1, 
	"S": 25, 
	"b": 4, 
	"bar_c": 4, 
	"rho": 1.3,
	"Cd0": 0.0434, 
	"Cda": 0.22,
	"Clb": -0.13, 
	"Clp": -0.505, 
	"Clr": 0.252, 
	"Clda": 0.0855, 
	"Cldr": -0.0024,
	"Cm0": 0.135, 
	"Cma": -1.50, 
	"Cmq": -38.2, 
	"Cmde": -0.992,
	"Cnb": 0.0726, 
	"Cnp": -0.069, 
	"Cnr": -0.0946, 
	"Cnda": 0.7, 
	"Cndr": -0.0693,
	"Cyb": -0.83, 
	"Cyp": 1, 
	"Cyr": 1, 
	"Cydr": 1,
	"Cz0": 0.23, 
	"Cza": 4.58, 
	"Czq": 1, 
	"Czde": 1,
	"Cx0": 1, 
	"Cxq": 1,
	"fault": fault,}

state = []
goal = []


fault_control_index = 1
fault_duration = 1000

fault_time = 0

def main():
	dynamics = FixedWing(x=x0, nominal_params=nominal_params, dt=dt, controller_dt=dt)
	NN_controller = NNController_new(n_state=9, m_control=4)
	NN_cbf = CBF(dynamics, n_state=9, m_control=4)
	NN_alpha = alpha_param(n_state=9)

	NN_controller.load_state_dict(torch.load('./data/FW_controller_NN_weights.pth'))
	NN_cbf.load_state_dict(torch.load('./data/FW_cbf_NN_weights.pth'))
	NN_alpha.load_state_dict(torch.load('./data/FW_alpha_NN_weights.pth'))

	NN_cbf.eval()
	NN_controller.eval()
	NN_alpha.eval()


	# NN_cbf_scripted = torch.jit.script(NN_cbf)
	# NN_cbf_trace = torch.jit.trace(NN_cbf)

	dummy_input = torch.ones(1,9)

	V, JV = NN_cbf.V_with_jacobian(dummy_input)
	print(100 * JV.float())

	NN_cbf_trace = torch.jit.trace(NN_cbf,dummy_input)

	# print(NN_cbf_trace.code)

	NN_cbf_trace.save("./torch2c/traced_CBF_model.pt")
	
	print(NN_cbf_trace(dummy_input))
		

if __name__ == '__main__':
	main()
