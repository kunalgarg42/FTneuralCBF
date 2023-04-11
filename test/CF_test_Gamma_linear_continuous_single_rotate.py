import os
import sys
import torch
import numpy as np
import argparse
import random
import tqdm

sys.path.insert(1, os.path.abspath('..'))
sys.path.insert(1, os.path.abspath('.'))

from pytictoc import TicToc
from dynamics.Crazyflie import CrazyFlies
from trainer import config
from trainer.constraints_crazy import constraints
from trainer.datagen import Dataset_with_Grad
from trainer.trainer import Trainer
from trainer.utils import Utils
from trainer.NNfuncgrad_CF import CBF, Gamma_linear

xg = torch.tensor([[0.0,
                    0.0,
                    5.5,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0]])

x0 = torch.tensor([[2.0,
                    2.0,
                    3.1,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0]])

dt = 0.001
n_state = 12
m_control = 4

R = torch.eye(n_state)

R[0][1] = 1.0
R[1][0] = -1.0
R[0][0] = 0.0
R[1][1] = 0.0
R[3][4] = 1.0
R[4][3] = -1.0
R[3][3] = 0.0
R[4][4] = 0.0
R[6][7] = 1.0
R[7][6] = -1.0
R[6][6] = 0.0
R[7][7] = 0.0
R[9][10] = 1.0
R[10][9] = -1.0
R[9][9] = 0.0
R[10][10] = 0.0

Rot_mat = torch.zeros(4, n_state, n_state)
Rot_mat[0, :, :] = torch.eye(n_state)
Rot_mat[1,:, :] = R.T
Rot_mat[2, :, :] = torch.matmul(Rot_mat[1, :, :], R).T
Rot_mat[3, :, :] = torch.matmul(Rot_mat[2, :, :], R).T

Rot_u = torch.tensor([[0.0, 1.0, 0.0, 0.0], 
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0],
                      [1.0, 0.0, 0.0, 0.0]]).T

Rot_u = torch.eye(4)

Rot_u_inv = torch.tensor([[0.0, 0.0, 0.0, 1.0], 
                      [1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0]]).T

Rot_u_inv = torch.eye(4)

nominal_params = config.CRAZYFLIE_PARAMS

fault = nominal_params["fault"]

use_good = 0

n_sample = 1000

traj_len = 100

Eval_steps = 299

fault = nominal_params["fault"]

t = TicToc()

gpu_id = 0

def main(args):
    fault_control_index = args.fault_index
    use_nom = args.use_nom
    str_data = './data/CF_gamma_NN_class_linear.pth'
    str_good_data = './good_data/data/CF_gamma_NN_class_linear.pth'
    dynamics = CrazyFlies(x=x0, goal=xg, nominal_params=nominal_params, dt=dt)
    util = Utils(n_state=n_state, m_control=m_control, dyn=dynamics, params=nominal_params, fault=fault,
                 fault_control_index=fault_control_index)
    cbf = CBF(dynamics=dynamics, n_state=n_state, m_control=m_control, fault=fault,
              fault_control_index=fault_control_index)
    gamma = Gamma_linear(n_state=n_state, m_control=m_control, traj_len=traj_len)

    if use_good == 1:
        try:
            gamma.load_state_dict(torch.load(str_good_data))
        except:
            gamma.load_state_dict(torch.load(str_data))
            print("No good data available")
    else:
        gamma.load_state_dict(torch.load(str_data))
    
    gamma.eval()
        
    cbf.load_state_dict(torch.load('./good_data/data/CF_cbf_NN_weightsCBF.pth'))
    cbf.eval()

    sm, sl = dynamics.state_limits()
    
    gamma_actual_bs = torch.ones(n_sample, m_control)

    # for j in range(n_sample):
    #     temp_var = np.mod(j, 2)
    #     if temp_var < 1:
    #         gamma_actual_bs[j, fault_control_index] = 0.0
        # if temp_var < 4:
        #     gamma_actual_bs[j, temp_var] = 0.0
    for j in range(n_sample):
            temp_var = np.mod(j, 5)
            if temp_var < 3:
                gamma_actual_bs[j, fault_control_index] = 0.0
                
    # rand_ind = torch.randperm(n_sample)
    # gamma_actual_bs = gamma_actual_bs[rand_ind, :]

    # state0 = (safe_m, safe_l, n_sample)
    state0 = dynamics.sample_safe(n_sample)
    # state0 = x0 + torch.randn(n_sample, n_state) * 10

    state_traj = torch.zeros(n_sample, Eval_steps, n_state)
    
    state_traj_diff = state_traj.clone()

    u_traj = torch.zeros(n_sample, Eval_steps, m_control)

    state_traj1 = torch.zeros(n_sample, Eval_steps, n_state)
    
    state_traj_diff1 = state_traj.clone()

    u_traj1 = torch.zeros(n_sample, Eval_steps, m_control)

    state_traj2 = torch.zeros(n_sample, Eval_steps, n_state)
    
    state_traj_diff2 = state_traj.clone()

    u_traj2 = torch.zeros(n_sample, Eval_steps, m_control)

    state_traj3 = torch.zeros(n_sample, Eval_steps, n_state)
    
    state_traj_diff3 = state_traj.clone()

    u_traj3 = torch.zeros(n_sample, Eval_steps, m_control)
    
    state = state0.clone()

    state1 = torch.matmul(state, Rot_mat[1, :, :])
    state2 = torch.matmul(state, Rot_mat[2, :, :])
    state3 = torch.matmul(state, Rot_mat[3, :, :])
    
    state_no_fault = state0.clone()

    state_no_fault1 = state1.clone()

    state_no_fault2 = state2.clone()

    state_no_fault3 = state3.clone()

    u_nominal = dynamics.u_nominal(state)
    
    t.tic()

    print('length of failure, acc fail , acc no fail')

    new_goal = dynamics.sample_safe(1)

    new_goal = new_goal.reshape(n_state, 1)

    acc_ind_temp = torch.zeros(1, 8)

    for k in range(Eval_steps):

        u_nominal = dynamics.u_nominal(state, op_point=new_goal)

        fx = dynamics._f(state, params=nominal_params)
        gx = dynamics._g(state, params=nominal_params)

        if use_nom == 0:
            h, grad_h = cbf.V_with_jacobian(state.reshape(n_sample, n_state, 1))
            u = util.fault_controller(u_nominal, fx, gx, h, grad_h)
        else:
            u = u_nominal.clone()

        u1 = torch.matmul(u, Rot_u)
        u2 = torch.matmul(u1, Rot_u)
        u3 = torch.matmul(u2, Rot_u)

        state_traj[:, k, :] = state.clone()
        
        state_traj_diff[:, k, :] = state_no_fault.clone() - state.clone()
        
        u_traj[:, k, :] = u.clone()

        state_traj1[:, k, :] = state1.clone()
        
        state_traj_diff1[:, k, :] = state_no_fault1.clone() - state1.clone()
        
        u_traj1[:, k, :] = u1.clone()

        state_traj1[:, k, :] = state2.clone()
        
        state_traj_diff2[:, k, :] = state_no_fault2.clone() - state2.clone()
        
        u_traj2[:, k, :] = u2.clone()

        state_traj3[:, k, :] = state3.clone()
        
        state_traj_diff3[:, k, :] = state_no_fault3.clone() - state3.clone()
        
        u_traj3[:, k, :] = u3.clone()

        gxu_no_fault = torch.matmul(gx, u.reshape(n_sample, m_control, 1))
        
        if k >= traj_len - 1:
            u = u * gamma_actual_bs
        
        gxu = torch.matmul(gx, u.reshape(n_sample, m_control, 1))

        dx = fx.reshape(n_sample, n_state) + gxu.reshape(n_sample, n_state)

        dx_no_fault = fx.reshape(n_sample, n_state) + gxu_no_fault.reshape(n_sample, n_state)
        
        state_no_fault = state.clone() + dx_no_fault * dt

        state_no_fault1 = state1.clone() + torch.matmul(dx_no_fault, Rot_mat[1, :, :]) * dt

        state_no_fault2 = state2.clone() + torch.matmul(dx_no_fault, Rot_mat[2, :, :]) * dt

        state_no_fault3 = state3.clone() + torch.matmul(dx_no_fault, Rot_mat[3, :, :]) * dt

        state = state.clone() + dx * dt
    
        for j2 in range(n_state):
            ind_sm = state[:, j2] > sm[j2]
            if torch.sum(ind_sm) > 0:
                state[ind_sm, j2] = sm[j2].repeat(torch.sum(ind_sm),)
            ind_sl = state[:, j2] < sl[j2]
            if torch.sum(ind_sl) > 0:
                state[ind_sl, j2] = sl[j2].repeat(torch.sum(ind_sl),)

        state1 = torch.matmul(state, Rot_mat[1, :, :])
        state2 = torch.matmul(state, Rot_mat[2, :, :])
        state3 = torch.matmul(state, Rot_mat[3, :, :])

        if k >= traj_len - 1:
            
            gamma_NN = gamma(state_traj[:, k - traj_len + 1:k + 1, :], state_traj_diff[:, k - traj_len + 1:k + 1, :], u_traj[:, k - traj_len + 1:k + 1, :])
            
            gamma_pred_NN = gamma_NN.reshape(n_sample, 1).clone().detach()

            gamma_pred = torch.ones(n_sample, m_control)

            gamma_pred1 = torch.ones(n_sample, m_control)

            gamma_pred2 = torch.ones(n_sample, m_control)

            gamma_pred3 = torch.ones(n_sample, m_control)

            gamma_pred[:, 0] = gamma_pred_NN[:, 0]

            gamma_pred1[:, 1] = gamma_pred_NN[:, 0]

            gamma_pred2[:, 2] = gamma_pred_NN[:, 0]

            gamma_pred3[:, 3] = gamma_pred_NN[:, 0]

            acc_ind = torch.zeros(1, m_control+1)

            acc_ind1 = torch.zeros(1, m_control+1)

            acc_ind2 = torch.zeros(1, m_control+1)

            acc_ind3 = torch.zeros(1, m_control+1)            
            
            index_fault = (gamma_actual_bs - 0.5) < 0
                
            index_num = torch.sum(index_fault.float())

            acc_ind_temp[0, 0] = torch.sum((gamma_pred[index_fault] < 0).float()) / (index_num + 1e-5)

            acc_ind_temp[0, 1] = torch.sum((gamma_pred1[index_fault] < 0).float()) / (index_num + 1e-5)

            acc_ind_temp[0, 2] = torch.sum((gamma_pred2[index_fault] < 0).float()) / (index_num + 1e-5)

            acc_ind_temp[0, 3] = torch.sum((gamma_pred3[index_fault] < 0).float()) / (index_num + 1e-5)

            index_no_fault = (gamma_actual_bs - 0.5) > 0

            index_num = torch.sum(index_no_fault.float())

            acc_ind_temp[0, 4] = torch.sum((gamma_pred[index_no_fault]> 0).float()) / (index_num + 1e-5)

            acc_ind_temp[0, 5] = torch.sum((gamma_pred1[index_no_fault]> 0).float()) / (index_num + 1e-5)

            acc_ind_temp[0, 6] = torch.sum((gamma_pred2[index_no_fault]> 0).float()) / (index_num + 1e-5)

            acc_ind_temp[0, 7] = torch.sum((gamma_pred3[index_no_fault]> 0).float()) / (index_num + 1e-5)

            print('{}, {}'.format(np.max([np.min([k - (traj_len - 2), traj_len]), 0]), acc_ind_temp.reshape(1, 8)[0].numpy()))
  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-fault_index', type=int, default=1)
    parser.add_argument('-use_nom', type=int, default=1)
    args = parser.parse_args()
    main(args)
