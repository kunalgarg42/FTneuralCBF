import os
import sys
sys.path.insert(1, os.path.abspath('.'))


import torch
import numpy as np
from sympy.solvers import nsolve
from sympy import Symbol, sin, cos
import matplotlib.pyplot as plt
from dynamics.fixed_wing_dyn import fw_dyn_ext, fw_dyn
from qp_control import config
from CBF import CBF
from qp_control.qp_con import QP_con
from qp_control.constraints_fw import constraints
import math

from qp_control.datagen import Dataset
from qp_control.trainer import Trainer

from qp_control.NNfuncgrad import CBF, alpha_param, NNController


x0 = [30, -1.5,-1.08, 1]

m = 23;

dt = 0.02
cla = 2*math.pi;
Cd0 = 0.02;
k1 = 0.05;
k2 = 0.05;

cm0 = -0.1;
cma = -0.1;
cmq = -9;
cmd = -1;
I = 4.51;
grav = 9.8;

rho = 1.225;
S = 0.99;
c = 0.33;

F = Symbol('F')
a = Symbol('a')

Vr = 25
gr = 0

f1 = -1/2*rho*(Vr**2)*S*(Cd0+cla*a+(cla**2)*(a**2))+F*cos(a)-m*grav*sin(gr)

f2 = 1/2*rho*(Vr**2)*S*(cla*a)+F*sin(a)-m*grav*cos(gr)

Ftr, alphar = nsolve((f1, f2),(F,a),(0,0))

state = []
obstacle= []
goal = []

def main():
	nn_controller = NNController(n_state=4, k_obstacle=1, m_control=2)
	# nn_controller.load_state_dict(torch.load('./data/drone_controller_weights.pth'))
	# nn_controller.eval()
	cbf = CBF(n_state=4, k_obstacle=1, m_control=2)
	# cbf.load_state_dict(torch.load('./data/drone_cbf_weights.pth'))
	# v_cbf = V_with_jacobian()
	alpha = alpha_param(n_state=4, k_obstacle=1)
	# alpha.load_state_dict(torch.load('./data/drone_alpha_weights.pth'))
	dataset = Dataset(n_state=4, k_obstacle=1, m_control=2, n_pos=1,safe_dist = 4, dang_dist = 2.5)
	trainer = Trainer(nn_controller, cbf, alpha, dataset, fw_dyn_ext, n_pos=1, dt = dt, safe_dist = 4, dang_dist = 2.5, action_loss_weight=0.1)
	state = x0
	obstacle = np.array([Vr+5, -1.5, -1.5, 1]).reshape(1,4)
	goal = [Vr, gr, gr+alphar,0]

	state_error = np.zeros((4,), dtype=np.float32)

	safety_rate = 0.0
	goal_reached = 0.0
	loss_total = 100.0

	for i in range(config.TRAIN_STEPS):
		# print(i)
		if np.mod(i, config.INIT_STATE_UPDATE) == 0 and i > 0:
			noise = np.random.normal(size=(1,)) * 0.1
		# 	# x = obstacle + np.array(noise).reshape(1,4)
			state[0] = obstacle[0] + np.array(noise)
		# 	# state = x
		# 	# print(x)
		if np.mod(i,2.0 * config.INIT_STATE_UPDATE) == 0 and i > 0:
			noise = np.random.normal(size=(4,)) * 0.1
			state = goal + np.array(noise).reshape(1,4)
			state = np.array(state,dtype = np.float32)
			# state = np.array(state).reshape(1,4)
		# print(i)

		state = np.array(state).reshape(1,4)
		if state[0][0]<0 or state[0][0]>45:
			state[0][0] = 10

		if state[0][1]<-1 or state[0][1]>1:
			state[0][1] = 0.5

		if state[0][2]<-1 or state[0][2]>1:
			state[0][2] = 0.25

		if state[0][3]<-2 or state[0][3]>2:
			state[0][3] = 1

		goal = np.array(goal).reshape(1,4)

		# print(state)

		u_nominal = Trainer.nominal_controller(state, goal, 5,fw_dyn,constraints)
		u_nominal = np.array(u_nominal)



		u = nn_controller(
            torch.from_numpy(state.reshape(1, 4).astype(np.float32)), 
            torch.from_numpy(obstacle.reshape(1, 4, 1).astype(np.float32)),
            torch.from_numpy(u_nominal.reshape(1, 2).astype(np.float32)),
            torch.from_numpy(state_error.reshape(1, 4).astype(np.float32)))
		u = np.squeeze(u.detach().cpu().numpy())

		# print(u.shape)
		# print(u_nominal.shape)

		norm_u = np.linalg.norm(u)
		
		if norm_u > 5:
			u = u / norm_u * 5
		
		fx, gx = fw_dyn(state.reshape(4,1))

		# if math.isnan(u[0]):
		# 	print(i)
		# 	print(state)
		# 	print(fx)
		# 	print(gx)
		# 	print(u)

		gxu = np.matmul(gx,u, dtype = object)
		gxu = np.array(gxu,dtype = float).reshape(1,4)
		dx = np.array(fx,dtype = float).reshape(1,4) + gxu
		# dx = float(dx)
		dx = dx.reshape(1,4)
		dx = np.array(dx, dtype = float)
		noise = Trainer.get_noise()
		
		x = np.array(state).reshape(1,4) + dx*dt + np.array(noise).reshape(1,4)*2
		state_next = x
        # state_next, state_nominal_next, obstacle_next, goal_next, done = env.step(u)

		dataset.add_data(state, obstacle, u_nominal, state_next, state_error)

		is_safe = int(Trainer.is_safe(state, obstacle, dang_dist= 2.5))
		safety_rate = safety_rate * (1 - 1e-4) + is_safe * 1e-4

		state_nominal_next = np.array(x).reshape(1,4) 
        # error between the true current state and the state obtained from the nominal model
        # this error will be fed to the controller network in the next timestep
		state_error = (state_next - state_nominal_next) / dt
		# print(i)
		# print(state_next)

		state = state_next
		# if math.isnan(state[0][0]):
			# print(x)
			# print(u)
			# print(i)
			# break
		done = np.linalg.norm(np.array(state_next,dtype = float) - np.array(goal,dtype= float)) < 1

        # obstacle = obstacle_next
        # goal = goal_next
		if np.mod(i, config.POLICY_UPDATE_INTERVAL) == 0 and i > 0:
			loss_np, acc_np, loss_h_safe, loss_h_dang, loss_alpha, loss_deriv_safe , loss_deriv_dang , loss_deriv_mid , loss_action = trainer.train_cbf_and_controller()
			print('step: {}, train h and u, loss: {:.3f}, safety rate: {:.3f}, goal reached: {:.3f}, acc: {}'.format(
                i, loss_np, safety_rate, goal_reached, acc_np))
			loss_total = loss_np
			# print(loss_h_safe.detach().cpu().numpy())
			# print(loss_h_dang.detach().cpu().numpy())
			# print(loss_alpha.detach().cpu().numpy())
			# print(loss_deriv_safe.detach().cpu().numpy())
			# print(loss_deriv_dang.detach().cpu().numpy())
			# print(loss_deriv_mid.detach().cpu().numpy())
			# print(loss_action.detach().cpu().numpy())
			# print(loss_np)
			# print(u)
			# print(u_nominal)


			torch.save(cbf.state_dict(), './data/drone_cbf_weights.pth')
			torch.save(nn_controller.state_dict(), './data/drone_controller_weights.pth')
			torch.save(alpha.state_dict(), './data/drone_alpha_weights.pth')

		if done:
			dist = np.linalg.norm(np.array(state_next,dtype = float) - np.array(goal,dtype= float))
			goal_reached = goal_reached * (1-1e-2) + (dist < 2.0) * 1e-2
			# print(goal_reached)
			state = x0
			# state, obstacle, goal = env.reset()
		# if loss_total<0.01:
		# 	print(i)
		# 	break


	

if __name__ == '__main__':
	main()
	

