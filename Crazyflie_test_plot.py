import os
import sys

sys.path.insert(1, os.path.abspath('.'))

import torch
import math
import random
import numpy as np
# from dynamics.fixed_wing_dyn import fw_dyn_ext, fw_dyn
from dynamics.Crazyflie import CrazyFlies
from qp_control import config
from CBF import CBF
from qp_control.constraints_crazy import constraints

from qp_control.datagen import Dataset_with_Grad
from qp_control.trainer_crazy import Trainer
from qp_control.utils_crazy import Utils

from qp_control.NNfuncgrad import CBF, alpha_param, NNController_new

import matplotlib.pyplot as plt

xg = torch.tensor([[2.0,
                    2.0,
                    2.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0]])

x0 = torch.tensor([[0.0,
                    0.0,
                    2.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0]])

dt = 0.01
n_state = 12
m_control = 4
fault = 1

nominal_params = {
    "m": 0.0299,
    "Ixx": 1.395 * 10**(-5),
    "Iyy": 1.395 * 10**(-5),
    "Izz": 2.173 * 10**(-5),
    "CT": 3.1582 * 10**(-10),
    "CD": 7.9379 * 10**(-12),
    "d": 0.03973,
    "fault": fault,}

state = []
goal = []

fault_control_index = 1
fault_duration = 100

fault_known = 1

def main():
	dynamics = CrazyFlies(x=x0, nominal_params=nominal_params, dt=dt, controller_dt=dt)
	util = Utils(n_state=n_state, m_control = m_control, dyn = dynamics, params = nominal_params, fault = fault, fault_control_index = fault_control_index)
	

	NN_controller = NNController_new(n_state=n_state, m_control=m_control)
	NN_cbf = CBF(dynamics, n_state=n_state, m_control=m_control)
	NN_alpha = alpha_param(n_state=n_state)

	NN_controller.load_state_dict(torch.load('./data/data/CF_controller_NN_weights.pth'))
	NN_cbf.load_state_dict(torch.load('./data/data/CF_cbf_NN_weights.pth'))
	NN_alpha.load_state_dict(torch.load('./data/data/CF_alpha_NN_weights.pth'))

	NN_cbf.eval()
	NN_controller.eval()
	NN_alpha.eval()


	FT_controller = NNController_new(n_state=n_state, m_control=m_control)
	FT_cbf = CBF(dynamics, n_state=n_state, m_control=m_control)
	FT_alpha = alpha_param(n_state=n_state)

	FT_controller.load_state_dict(torch.load('./data/data/CF_controller_FT_weights.pth'))
	FT_cbf.load_state_dict(torch.load('./data/data/CF_cbf_FT_weights.pth'))
	FT_alpha.load_state_dict(torch.load('./data/data/CF_alpha_FT_weights.pth'))

	FT_cbf.eval()
	FT_controller.eval()
	FT_alpha.eval()

	state = x0
	goal = xg
	goal = np.array(goal).reshape(1,n_state)

	safety_rate = 0
	goal_reached = 0
	num_episodes = 0
	traj_following_error = 0
	epsilon = 0.1

	um, ul = dynamics.control_limits()

	sm, sl = dynamics.state_limits()

	x_pl = np.array(state).reshape(1,n_state)
	u_pl = np.array([0]*m_control).reshape(1,m_control)
	h, _ = NN_cbf.V_with_jacobian(state.reshape(1,n_state,1))

	# print(h)
	h_pl = np.array(h.detach()).reshape(1,1)

	rand_start = random.uniform(1.01,100)

	fault_start_epoch = math.floor(config.EVAL_STEPS / rand_start) + 1000000
	fault_start = 0
	for i in range(config.EVAL_STEPS):
		# print(i)

		for j in range(n_state):
			if state[0,j] < 0.5 * sl[j]:
				state[0,j] = sl[j]
			if state[0,j] > 2 * sm[j]:
				state[0,j] = sm[j]
		
		
		fx = dynamics._f(state , params = nominal_params)
		gx = dynamics._g(state , params = nominal_params)
		
		u_nominal = util.nominal_controller(state = state, goal = goal, u_norm_max = 5, dyn = dynamics, constraints = constraints)


		for j in range(m_control):
			if u_nominal[0,j] < ul[j]:
				u_nominal[0,j] = ul[j]
			if u_nominal[0,j] > um[j]:
				u_nominal[0,j] = um[j]

		
		if fault_known == 1:
			## 1 -> time-based switching, assumes knowledge of when fault occurs and stops
			## 0 -> Fault-detection based-switching, using the proposed scheme from the paper
			
			if fault_start == 0 and i >= fault_start_epoch and i <= fault_start_epoch + fault_duration and util.is_safe(state):
				fault_start = 1

			if fault_start == 1 and i > fault_start_epoch + fault_duration:
				fault_start == 0

			if fault_start == 1:
				h, grad_h = NN_cbf.V_with_jacobian(state.reshape(1,n_state,1))
			else:
				h, grad_h = FT_cbf.V_with_jacobian(state.reshape(1,n_state,1))
			
			u = util.neural_controller(u_nominal, fx, gx, h, grad_h)		
			u = torch.squeeze(u.detach().cpu())

			if fault_start == 1:
				u[fault_control_index] = torch.rand(1) * 5
			
			# for j in range(m_control):
			# 	if u[j] <= ul[j]:
			# 		u[j] = um[j]/2
			# 	if u[j] > um[j]:
			# 		u[j] = um[j]

			if torch.isnan(torch.sum(u)):
				i = i-1
				continue

			u = torch.tensor(u, dtype = torch.float32)
			gxu = torch.matmul(gx,u.reshape(m_control,1))

			dx = fx.reshape(1,n_state) + gxu.reshape(1,n_state)
	
		else:
			h, grad_h = NN_cbf.V_with_jacobian(state.reshape(1,n_state,1))
			u = util.neural_controller(u_nominal, fx, gx, h, grad_h)		
			u = torch.squeeze(u.detach().cpu())

			if i >= fault_start_epoch and i <= fault_start_epoch + fault_duration:
				u[fault_control_index] = torch.rand(1)
			
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

			dot_h = torch.matmul(dx, grad_h.reshape(n_state,1))
			if dot_h < epsilon:
				h, grad_h = FT_cbf.V_with_jacobian(state.reshape(1,n_state,1))
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
		
		x_pl = np.vstack((x_pl,np.array(state.clone().detach()).reshape(1,n_state)))
		u_pl = np.vstack((u_pl,np.array(u.clone().detach()).reshape(1,m_control)))
		h_pl = np.vstack((h_pl, np.array(h.clone().detach()).reshape(1,1)))
	
	time_pl = np.arange(0., dt * config.EVAL_STEPS + dt, dt)

	z_pl = x_pl[:,2]
	# print(x_alpha.shape)
	u1 = u_pl[:,0]
	u2 = u_pl[:,1]
	u3 = u_pl[:,2]
	u4 = u_pl[:,3]
	# print(u1.shape)

	# print(h_pl.shape)

	plt.figure(figsize=(9, 3))

	plt.subplot(131)
	plt.plot(time_pl, z_pl)
	plt.subplot(132)
	plt.plot(time_pl,h_pl)
	plt.subplot(133)
	plt.plot(time_pl, u1, time_pl, u2, time_pl, u3, time_pl, u4)
	# plt.suptitle('Categorical Plotting')
	plt.show()
if __name__ == '__main__':
	main()
	


# scp -r kgarg@18.18.47.27:/home/kgarg/kunal_files/MIT_REALM/fault_tol_control/data/data /home/kunal/MIT_REALM/Research/fault_tol_control/data/