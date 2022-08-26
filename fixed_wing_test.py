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
fault_duration = 1000

def main():
	dynamics = FixedWing(x = x0, nominal_params = nominal_params, dt = dt, controller_dt= dt)
	util = Utils(n_state=9, m_control = 4, dyn = dynamics, params = nominal_params, fault = fault, fault_control_index = fault_control_index)
	
	NN_controller = NNController_new(n_state=9, m_control=4)
	NN_cbf = CBF(n_state=9, m_control=4)
	NN_alpha = alpha_param(n_state=9)

	NN_controller.load_state_dict(torch.load('./data/FW_controller_NN_weights.pth'))
	NN_cbf.load_state_dict(torch.load('./data/FW_cbf_NN_weights.pth'))
	NN_alpha.load_state_dict(torch.load('./data/FW_alpha_NN_weights.pth'))

	NN_cbf.eval()
	NN_controller.eval()
	NN_alpha.eval()


	FT_controller = NNController_new(n_state=9, m_control=4)
	FT_cbf = CBF(n_state=9, m_control=4)
	FT_alpha = alpha_param(n_state=9)

	FT_controller.load_state_dict(torch.load('./data/FW_controller_FT_weights.pth'))
	FT_cbf.load_state_dict(torch.load('./data/FW_cbf_FT_weights.pth'))
	FT_alpha.load_state_dict(torch.load('./data/FW_alpha_FT_weights.pth'))

	FT_cbf.eval()
	FT_controller.eval()
	FT_alpha.eval()

	state = x0
	goal = xg
	goal = np.array(goal).reshape(1,9)

	safety_rate = 0
	goal_reached = 0
	num_episodes = 0
	traj_following_error = 0

	um, ul = dynamics.control_limits()

	sm, sl = dynamics.state_limits()
	for num_epoch in range(config.EVAL_EPOCHS):
		
		rand_start = random.uniform(1.01,100)

		fault_start_epoch = math.floor(config.EVAL_STEPS / rand_start)

		for i in range(config.EVAL_STEPS):

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

			if i < fault_start_epoch or i> fault_start_epoch + fault_duration:
				h, grad_h = NN_cbf.V_with_jacobian(state.reshape(1,9,1))
			else:
				h, grad_h = FT_cbf.V_with_jacobian(state.reshape(1,9,1))
				
			u = util.neural_controller(u_nominal, fx, gx, h, grad_h)		
			u = torch.squeeze(u.detach().cpu())

			if i >= fault_start_epoch and i <= fault_start_epoch + fault_duration:
				u[fault_control_index] = torch.rand(1) * 5
			
			for j in range(m_control):
				if u[j] < ul[j]:
					u[j] = ul[j]
				if u[j] > um[j]:
					u[j] = um[j]

			if torch.isnan(torch.sum(u)):
				i = i-1
				continue

			u = torch.tensor(u, dtype = torch.float32)
			gxu = torch.matmul(gx,u.reshape(m_control,1))

			dx = fx.reshape(1,n_state) + gxu.reshape(1,n_state)

			state_next = state + dx*dt

			is_safe = int(util.is_safe(state))
			safety_rate = safety_rate * (1 - 1e-4) + is_safe * 1e-4

			state = state_next.clone()
			dist = torch.linalg.norm(state_next.detach().cpu() - goal)
			done =  dist < 5

			traj_following_error = traj_following_error * i / (i+1) + dist / (i+1)

			if done:
				num_episodes = num_episodes + 1
				goal_reached = goal_reached + 1 if dist < goal_radius else goal_reached
				print('Progress: {:.2f}% safety rate: {:.4f}, distance: {:.4f}'.format(
					100 * (num_epoch + 1.0) / config.EVAL_EPOCHS, safety_rate, dist))
				state = x0 + torch.rand(1,n_state)
				continue

if __name__ == '__main__':
	main()
	