import os
import sys
sys.path.insert(1, os.path.abspath('.'))


import torch
import numpy as np
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
from qp_control.trainer_new import Trainer
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
fault = 1

nominal_params = {
	"m": 1.0, 
	"g": 9.8, 
	"Ixx": 1, 
	"Iyy": 1, 
	"Izz": 1, 
	"Ixz": 0.1, 
	"S": 1, 
	"b": 1, 
	"bar_c": 1, 
	"rho": 1,
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

def main():
	dynamics = FixedWing(x = x0, nominal_params = nominal_params, dt = dt, controller_dt= dt)
	util = Utils(n_state=9, m_control = 4, j_const = 2, dyn = dynamics, dt = dt, params = nominal_params, fault = fault, fault_control_index = fault_control_index)
	nn_controller = NNController_new(n_state=9, m_control=4)
	cbf = CBF(n_state=9, m_control=4)
	alpha = alpha_param(n_state=9)
	dataset = Dataset_with_Grad(n_state=9, m_control=4, n_pos=1,safe_alpha = 0.3, dang_alpha = 0.4)
	trainer = Trainer(nn_controller, cbf, alpha, dataset, n_state=9, m_control = 4, j_const = 2, dyn = dynamics, n_pos=1, dt = dt, safe_alpha = 0.3, dang_alpha = 0.4, action_loss_weight=0.1, params = nominal_params, fault = fault, fault_control_index = fault_control_index)
	state = x0
	goal = xg
	goal = np.array(goal).reshape(1,9)

	state_error = torch.zeros(1, 9)

	safety_rate = 0.0
	goal_reached = 0.0
	loss_total = 100.0

	um, ul = dynamics.control_limits()

	sm, sl = dynamics.state_limits()

	for i in range(config.TRAIN_STEPS):
		print(i)
		if np.mod(i, config.INIT_STATE_UPDATE) == 0 and i > 0:
			state[0,1] = sm[1] + torch.rand(1) * 0.01

		for j in range(n_state):
			if state[0,j] < -1.0e1:
				state[0,j] = x0[0,0]
			if state[0,j] > 1.0e4:
				state[0,j] = sm[j]
		
		
		fx = dynamics._f(state , params = nominal_params)
		gx = dynamics._g(state , params = nominal_params)
		
		u_nominal = util.nominal_controller(state = state, goal = goal, u_norm_max = 5, dyn = dynamics, constraints = constraints)


		for j in range(m_control):
			if u_nominal[0,j] < ul[j]:
				u_nominal[0,j] = ul[j]
			if u_nominal[0,j] > um[j]:
				u_nominal[0,j] = um[j]

		u = nn_controller(torch.tensor(state,dtype = torch.float32), torch.tensor(u_nominal,dtype = torch.float32))

		u = torch.squeeze(u.detach().cpu())

		if fault == 1:
			u[fault_control_index] = torch.rand(1) * 5

		if torch.isnan(torch.sum(u)):
			i = i-1
			continue

		gxu = torch.matmul(gx,u.reshape(m_control,1))

		dx = fx.reshape(1,n_state) + gxu.reshape(1,n_state)

		state_next = state + dx*dt

		h, _ = cbf.V_with_jacobian(state.reshape(1,9,1))

		dataset.add_data(state, u, u_nominal)

		is_safe = int(util.is_safe(state))
		safety_rate = safety_rate * (1 - 1e-4) + is_safe * 1e-4

		state = state_next.clone()
		done = torch.linalg.norm(state_next.detach().cpu() - goal) < 5

		if np.mod(i, config.POLICY_UPDATE_INTERVAL) == 0 and i > 0:
			loss_np, acc_np, loss_h_safe, loss_h_dang, loss_alpha, loss_deriv_safe , loss_deriv_dang , loss_deriv_mid , loss_action = trainer.train_cbf_and_controller()
			print('step: {}, train h and u, loss: {:.3f}, safety rate: {:.3f}, goal reached: {:.3f}, acc: {}'.format(
                i, loss_np, safety_rate, goal_reached, acc_np))
			loss_total = loss_np

			if fault == 0:
				torch.save(cbf.state_dict(), './data/FW_cbf_NN_weights.pth')
				torch.save(nn_controller.state_dict(), './data/FW_controller_NN_weights.pth')
				torch.save(alpha.state_dict(), './data/FW_alpha_NN_weights.pth')
			else:
				torch.save(cbf.state_dict(), './data/FW_cbf_FT_weights.pth')
				torch.save(nn_controller.state_dict(), './data/FW_controller_FT_weights.pth')
				torch.save(alpha.state_dict(), './data/FW_alpha_FT_weights.pth')
		if done:
			dist = np.linalg.norm(np.array(state_next,dtype = float) - np.array(goal,dtype= float))
			goal_reached = goal_reached * (1-1e-2) + (dist < 2.0) * 1e-2
			state = x0

if __name__ == '__main__':
	main()
	
