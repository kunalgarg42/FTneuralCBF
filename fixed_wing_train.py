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

from qp_control.NNfuncgrad import CBF, alpha_param, NNController_new


# import cProfile
# cProfile.run('foo()')



x0 = torch.tensor([[50.0, 
0.0, 
0.0, 
0.0, 
0.0, 
0.0,
0.0,
0.0,
0.0]])


xg = torch.tensor([[100.0, 
0.2, 
0.0, 
0.0, 
0.0, 
0.0,
0.0,
0.0,
0.0]])


dt = 0.01
n_state = 9
m_control = 4

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
	"Cxq": 1,}

state = []
goal = []

def main():
	dynamics = FixedWing(x = x0, nominal_params = nominal_params, dt = dt, controller_dt= dt)
	nn_controller = NNController_new(n_state=9, m_control=4)
	# nn_controller.load_state_dict(torch.load('./data/drone_controller_weights.pth'))
	# nn_controller.eval()
	cbf = CBF(n_state=9, m_control=2)
	# cbf.load_state_dict(torch.load('./data/drone_cbf_weights.pth'))
	# v_cbf = V_with_jacobian()
	alpha = alpha_param(n_state=9)
	# alpha.load_state_dict(torch.load('./data/drone_alpha_weights.pth'))
	dataset = Dataset_with_Grad(n_state=9, m_control=4, n_pos=1,safe_alpha = 0.3, dang_alpha = 0.4)
	trainer = Trainer(nn_controller, cbf, alpha, dataset, n_state=9, m_control = 4, j_const = 2, dyn = dynamics, n_pos=1, dt = dt, safe_alpha = 0.3, dang_alpha = 0.4, action_loss_weight=0.1, params = nominal_params)
	state = x0
	goal = xg

	state_error = torch.zeros(1, 9)

	safety_rate = 0.0
	goal_reached = 0.0
	loss_total = 100.0

	for i in range(config.TRAIN_STEPS):
		print(i)

		
		goal = np.array(goal).reshape(1,9)

		# print(state)

		fx = dynamics._f(state ,params = nominal_params)
		gx = dynamics._g(state ,params = nominal_params)
		
		# print(fx)
		# print(fadsad)

		u_nominal = trainer.nominal_controller(state = state, goal = goal, u_norm_max = 5, dyn = dynamics, constraints = constraints)

		# u_nominal = np.array(u_nominal)

		u = nn_controller(torch.tensor(state,dtype = torch.float32), torch.tensor(u_nominal,dtype = torch.float32))



		u = torch.squeeze(u.detach().cpu())

		norm_u = torch.linalg.norm(u)
		
		if norm_u > 5:
			u = u / norm_u * 5

		gxu = torch.matmul(gx,u.reshape(m_control,1))


		dx = fx.reshape(1,n_state) + gxu.reshape(1,n_state)

		x = state + dx*dt

		# h = cbf(x)

		h, grad_h = cbf.V_with_jacobian(x.reshape(1,9,1))



		# print(h1)

		# print(Jh1)

		# print(Asasas)

		# grad_h = torch.autograd.grad(h,x,allow_unused=True,retain_graph = True)
		# h.retain_grad()
		# grad_h = torch.cat(grad_h, dim =0)

		# print(grad_h)
		# print(grad_h.shape)
		# print(state.shape)

		state_next = x

		dataset.add_data(grad_h, state, u, u_nominal, state_next)

		is_safe = int(trainer.is_safe(state))
		safety_rate = safety_rate * (1 - 1e-4) + is_safe * 1e-4

		state_nominal_next = x 

		state = state_next
		done = torch.linalg.norm(state_next.detach().cpu() - goal) < 1

        # obstacle = obstacle_next
        # goal = goal_next
		if np.mod(i, config.POLICY_UPDATE_INTERVAL) == 0 and i > 0:
			loss_np, acc_np, loss_h_safe, loss_h_dang, loss_alpha, loss_deriv_safe , loss_deriv_dang , loss_deriv_mid , loss_action = trainer.train_cbf_and_controller()
			print('step: {}, train h and u, loss: {:.3f}, safety rate: {:.3f}, goal reached: {:.3f}, acc: {}'.format(
                i, loss_np, safety_rate, goal_reached, acc_np))
			loss_total = loss_np


			torch.save(cbf.state_dict(), './data/drone_cbf_weights.pth')
			torch.save(nn_controller.state_dict(), './data/drone_controller_weights.pth')
			torch.save(alpha.state_dict(), './data/drone_alpha_weights.pth')

		if done:
			dist = np.linalg.norm(np.array(state_next,dtype = float) - np.array(goal,dtype= float))
			goal_reached = goal_reached * (1-1e-2) + (dist < 2.0) * 1e-2
			state = x0



	

if __name__ == '__main__':
	main()
	

