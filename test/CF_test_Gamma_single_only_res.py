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
from trainer.NNfuncgrad_CF import CBF, Gamma

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

n_sample = 1000

traj_len = 100

Eval_steps = 299

fault = nominal_params["fault"]

t = TicToc()

gpu_id = 0

def main(args):
    fault_control_index = args.fault_index
    use_nom = args.use_nom
    str_data = './data/CF_gamma_NN_weights_only_res.pth'
    str_good_data = './good_data/data/CF_gamma_NN_weights_only_res.pth'
    dynamics = CrazyFlies(x=x0, goal=xg, nominal_params=nominal_params, dt=dt)
    util = Utils(n_state=n_state, m_control=m_control, dyn=dynamics, params=nominal_params, fault=fault,
                 fault_control_index=fault_control_index)
    cbf = CBF(dynamics=dynamics, n_state=n_state, m_control=m_control, fault=fault,
              fault_control_index=fault_control_index)
    gamma = Gamma(n_state=n_state, m_control=m_control, traj_len=traj_len)

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
    safe_m, safe_l = dynamics.safe_limits(sm, sl, fault)
    
    gamma_actual_bs = torch.ones(n_sample, m_control)

    for j in range(n_sample):
        temp_var = np.mod(j, 2)
        if temp_var < 1:
            gamma_actual_bs[j, fault_control_index] = 0.0
    # gamma_actual_bs[:, 1] = torch.zeros(n_sample,)

    rand_ind = torch.randperm(n_sample)
    gamma_actual_bs = gamma_actual_bs[rand_ind, :]

    # state0 = (safe_m, safe_l, n_sample)
    state0 = dynamics.sample_safe(n_sample)
    
    state_traj = torch.zeros(n_sample, Eval_steps, n_state)
    
    state_traj_diff = state_traj.clone()

    u_traj = torch.zeros(n_sample, Eval_steps, m_control)
    
    state = state0.clone()
    
    state_no_fault = state0.clone()

    u_nominal = dynamics.u_nominal(state)
    
    t.tic()

    print('length of failure, acc fail , acc no fail')

    for k in range(Eval_steps):

        u_nominal = dynamics.u_nominal(state)

        fx = dynamics._f(state, params=nominal_params)
        gx = dynamics._g(state, params=nominal_params)

        if use_nom == 0:
            h, grad_h = cbf.V_with_jacobian(state.reshape(n_sample, n_state, 1))
            u = util.fault_controller(u_nominal, fx, gx, h, grad_h)
        else:
            u = u_nominal.clone()
                
        state_traj[:, k, :] = state.clone()
        
        state_traj_diff[:, k, :] = state_no_fault.clone() - state.clone()
        
        u_traj[:, k, :] = u.clone()

        gxu_no_fault = torch.matmul(gx, u.reshape(n_sample, m_control, 1))
        
        if k >= traj_len - 2:
            u = u * gamma_actual_bs
        
        gxu = torch.matmul(gx, u.reshape(n_sample, m_control, 1))

        dx = fx.reshape(n_sample, n_state) + gxu.reshape(n_sample, n_state)

        dx_no_fault = fx.reshape(n_sample, n_state) + gxu_no_fault.reshape(n_sample, n_state)
        
        state_no_fault = state.clone() + dx_no_fault * dt 

        state = state.clone() + dx * dt #  + torch.randn(n_sample, n_state) * dt

        for j2 in range(n_state):
            ind_sm = state[:, j2] > sm[j2]
            if torch.sum(ind_sm) > 0:
                state[ind_sm, j2] = sm[j2].repeat(torch.sum(ind_sm),)
            ind_sl = state[:, j2] < sl[j2]
            if torch.sum(ind_sl) > 0:
                state[ind_sl, j2] = sl[j2].repeat(torch.sum(ind_sl),)
        
        if k >= traj_len - 1:
            
            gamma_NN = gamma(0 * state_traj[:, k - traj_len + 1:k + 1, :], state_traj_diff[:, k - traj_len + 1:k + 1, :], 0 * u_traj[:, k - traj_len + 1:k + 1, :])
            
            gamma_pred = gamma_NN.reshape(n_sample, m_control).clone().detach()
            
            acc_ind = torch.zeros(1, m_control+1)            
            
            for j in range(m_control):
                index_fault = gamma_actual_bs[:, j]==0
                index_num = torch.sum(index_fault == True)
                acc_ind[0, j] = 1 - torch.abs(torch.sum(gamma_actual_bs[index_fault, j] - gamma_pred[index_fault, j]) / index_num)
            
            index_no_fault = torch.sum(gamma_actual_bs, dim=1) == m_control
            
            index_num = torch.sum(index_no_fault == True)
            
            acc_ind[0, -1] = torch.sum(gamma_pred[index_no_fault, :]) / (index_num + 1e-5) / m_control
            
            print('{}, {:.3f}, {:.3f}'.format(np.min([k - (traj_len - 2), traj_len]), acc_ind[0][1], acc_ind[0][-1]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-fault_index', type=int, default=1)
    parser.add_argument('-use_nom', type=int, default=1)
    args = parser.parse_args()
    main(args)
