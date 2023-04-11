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

use_good = 1

n_sample = 1000

traj_len = 100

Eval_steps = int(traj_len * 1.5)

fault = nominal_params["fault"]

t = TicToc()

gpu_id = 0

def main(args):
    fault_control_index = args.fault_index
    str_data = './data/CF_gamma_NN_weights{}.pth'.format(fault_control_index)
    str_good_data = './good_data/data/CF_gamma_NN_weights{}.pth'.format(fault_control_index)
    dynamics = CrazyFlies(x=x0, goal=xg, nominal_params=nominal_params, dt=dt)
    util = Utils(n_state=n_state, m_control=m_control, dyn=dynamics, params=nominal_params, fault=fault,
                 fault_control_index=fault_control_index)
    cbf = CBF(dynamics=dynamics, n_state=n_state, m_control=m_control, fault=fault,
              fault_control_index=fault_control_index)
    gamma = Gamma(n_state=n_state, m_control=m_control, traj_len=traj_len)

    if use_good == 1:
    # try:
        gamma.load_state_dict(torch.load(str_good_data))
    # except:
    #     print("No good data available")
    else:
        gamma.load_state_dict(torch.load(str_data))
    
    gamma.eval()
        
    cbf.load_state_dict(torch.load('./good_data/data/CF_cbf_NN_weightsCBF.pth'))
    cbf.eval()

    sm, sl = dynamics.state_limits()
    safe_m, safe_l = dynamics.safe_limits(sm, sl, fault)
    
    gamma_actual_bs = torch.ones(n_sample, m_control)
    fault_value = 0.0
    fault_index_extra = random.randint(0, m_control-1)
    for j in range(n_sample):
        fault_control_index = np.mod(j, 6)
        if fault_control_index < m_control:
            gamma_actual_bs[j, fault_control_index] = 0.0
        if fault_control_index == 5:
            gamma_actual_bs[j, fault_index_extra] = 0.0

    rand_ind = torch.randperm(n_sample)
    gamma_actual_bs = gamma_actual_bs[rand_ind, :]

    state0 = util.x_samples(safe_m, safe_l, n_sample)
    
    state_traj = torch.zeros(n_sample, Eval_steps, n_state)
    
    state_traj_diff = state_traj.clone()

    u_traj = torch.zeros(n_sample, Eval_steps, m_control)
    
    state = state0.clone()
    
    state_no_fault = state0.clone()

    u_nominal = dynamics.u_nominal(state)
    
    t.tic()

    for k in tqdm.trange(config.EVAL_STEPS):
        
        u_nominal = dynamics.u_nominal(state)

        fx = dynamics._f(state, params=nominal_params)
        gx = dynamics._g(state, params=nominal_params)

        # h, grad_h = cbf.V_with_jacobian(state.reshape(n_sample, n_state, 1))
        u = u_nominal.clone()
        # u = util.fault_controller(u_nominal, fx, gx, h, grad_h)

        state_traj[:, k, :] = state.clone()
        
        state_traj_diff[:, k, :] = state_no_fault.clone() - state.clone()
        
        u_traj[:, k, :] = u.clone()

        gxu_no_fault = torch.matmul(gx, u.reshape(n_sample, m_control, 1))
        
        u = u * gamma_actual_bs
        
        gxu = torch.matmul(gx, u.reshape(n_sample, m_control, 1))

        dx = fx.reshape(n_sample, n_state) + gxu.reshape(n_sample, n_state)

        dx_no_fault = fx.reshape(n_sample, n_state) + gxu_no_fault.reshape(n_sample, n_state)
        
        state_no_fault = state.clone() + dx_no_fault * dt 

        state = state.clone() + dx * dt + torch.randn(n_sample, n_state) * dt

        for j2 in range(n_state):
            ind_sm = state[:, j2] > sm[j2]
            if torch.sum(ind_sm) > 0:
                state[ind_sm, j2] = sm[j2].repeat(torch.sum(ind_sm),)
            ind_sl = state[:, j2] < sl[j2]
            if torch.sum(ind_sl) > 0:
                state[ind_sl, j2] = sl[j2].repeat(torch.sum(ind_sl),)
        
        if k >= traj_len - 1:
            # if np.mod(k + 1, traj_len) > 0:
            gamma_NN = gamma(state_traj[:, k - traj_len + 1:k + 1, :], (1 - args.fault_index) * state_traj_diff[:, k - traj_len + 1:k + 1, :], u_traj[:, k - traj_len + 1:k + 1, :])
            
            gamma_pred = gamma_NN.reshape(n_sample, m_control).clone().detach()
            for j in range(n_sample):
                min_gamma = int(torch.argmin(gamma_pred[j, :]))
                # print(min_gamma)
                for i in range(m_control):
                    if i == min_gamma and gamma_pred[j, min_gamma] < fault_value + 0.5:
                        gamma_pred[j, i] = 0.0                        
                    else:
                        gamma_pred[j, i] = 1.0
            acc_ind = torch.zeros(1, m_control+1)            
            for j in range(m_control):
                index_fault = gamma_actual_bs[:, j]==0
                index_num = torch.sum(index_fault)
                acc_ind[0, j] = 1 - torch.abs(torch.sum(gamma_actual_bs[index_fault, j] - gamma_pred[index_fault, j]) / (index_num + 1e-5))
            
            index_no_fault = torch.sum(gamma_actual_bs, dim=1) == m_control
            index_num = torch.sum(index_no_fault)
            acc_ind[0, -1] = torch.sum(gamma_pred[index_no_fault, :]) / (index_num + 1e-5) / m_control

            print(acc_ind)
            # print(gamma_pred)
            # print(gamma_actual_bs)
            # gamma_err = gamma_pred - gamma_actual_bs
            # correct_gamma1 = 1 - torch.sum(torch.abs(gamma_err[:, 0])) / n_sample
            # print(correct_gamma1)
            # correct_gamma2 = 1 - torch.sum(torch.abs(gamma_err[:, 1])) / n_sample
            # print(correct_gamma2)
            # correct_gamma3 = 1 - torch.sum(torch.abs(gamma_err[:, 2])) / n_sample
            # print(correct_gamma3)
            # correct_gamma4 = 1 - torch.sum(torch.abs(gamma_err[:, 3])) / n_sample
            # print(correct_gamma4)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-fault_index', type=int, default=0)
    args = parser.parse_args()
    main(args)
