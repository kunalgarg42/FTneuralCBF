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

nominal_params = config.CRAZYFLIE_PARAMS

fault = nominal_params["fault"]

use_good = 0

traj_len = 100

Eval_steps = 199

fault = nominal_params["fault"]

t = TicToc()

gpu_id = 1

def main(args):
    fault_control_index = args.fault_index
    use_nom = args.use_nom

    if use_nom == 1:
        n_sample = 5000
    else:
        n_sample = 1000
    str_data = './data_from_supercloud/CF_gamma_NN_class_linear_ALL_faults_no_res.pth'
    str_good_data = './good_data/data/CF_gamma_NN_class_linear_ALL_faults_no_res.pth'
    
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

    for j in range(n_sample):
            temp_var = np.mod(j, 5)
            if temp_var < 4:
                gamma_actual_bs[j, temp_var] = 0.0

    rand_ind = torch.randperm(n_sample)
    gamma_actual_bs = gamma_actual_bs[rand_ind, :]

    state0 = dynamics.sample_safe(n_sample)

    state_traj = torch.zeros(n_sample, Eval_steps, n_state)    

    u_traj = torch.zeros(n_sample, Eval_steps, m_control)
    
    state = state0.clone()
    
    u_nominal = dynamics.u_nominal(state)
    
    t.tic()

    print('length of failure, acc fail , acc no fail')

    new_goal = dynamics.sample_safe(1)

    new_goal = new_goal.reshape(n_state, 1)

    for k in range(Eval_steps):

        u_nominal = dynamics.u_nominal(state, op_point=new_goal)

        fx = dynamics._f(state, params=nominal_params)
        gx = dynamics._g(state, params=nominal_params)

        if use_nom == 0:
            h, grad_h = cbf.V_with_jacobian(state.reshape(n_sample, n_state, 1))
            u = util.fault_controller(u_nominal, fx, gx, h, grad_h)
        else:
            u = u_nominal.clone()

        state_traj[:, k, :] = state.clone()
                
        u_traj[:, k, :] = u.clone()
        
        if k >= traj_len - 1:
            u = u * gamma_actual_bs
        
        gxu = torch.matmul(gx, u.reshape(n_sample, m_control, 1))

        dx = fx.reshape(n_sample, n_state) + gxu.reshape(n_sample, n_state)

        state = state.clone() + dx * dt
    
        for j2 in range(n_state):
            ind_sm = state[:, j2] > sm[j2]
            if torch.sum(ind_sm) > 0:
                state[ind_sm, j2] = sm[j2].repeat(torch.sum(ind_sm),)
            ind_sl = state[:, j2] < sl[j2]
            if torch.sum(ind_sl) > 0:
                state[ind_sl, j2] = sl[j2].repeat(torch.sum(ind_sl),)

        if k >= traj_len - 1:
            
            gamma_NN = gamma(state_traj[:, k - traj_len + 1:k + 1, :], 0 * state_traj[:, k - traj_len + 1:k + 1, :], u_traj[:, k - traj_len + 1:k + 1, :])

            gamma_pred = gamma_NN.reshape(n_sample, m_control).clone().detach()

            acc_ind = torch.zeros(1, m_control * 2)

            for j in range(m_control):
                
                index_fault = gamma_actual_bs[:, j] < 0.5

                index_num = torch.sum(index_fault.float())

                if index_num > 0:
                    acc_ind[0, j] =  torch.sum((gamma_pred[index_fault, j] < 0).float()) / (index_num + 1e-5)
                else:
                    acc_ind[0, j] = 1
                    
                # if k == traj_len - 1:
                #     index_no_fault = torch.ones(n_sample,) > 0.5
                # else:
                index_no_fault = gamma_actual_bs[:, j] > 0.5
                
                index_num = torch.sum(index_no_fault.float())

                if index_num > 0:
                    acc_ind[0, j + m_control] =  torch.sum((gamma_pred[index_no_fault, j] > 0).float()) / (index_num + 1e-5)
                else:
                    acc_ind[0, j + m_control] = 1
                
            
            np.set_printoptions(precision=3, suppress=True)

            print('{}, {}'.format(np.max([np.min([k - (traj_len - 2), traj_len]), 0]), acc_ind[0].numpy()))
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-fault_index', type=int, default=1)
    parser.add_argument('-use_nom', type=int, default=1)
    args = parser.parse_args()
    main(args)
