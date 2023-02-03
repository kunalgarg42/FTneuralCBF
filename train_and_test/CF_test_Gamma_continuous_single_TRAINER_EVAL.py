import os
import sys
import torch
import numpy as np
import argparse
import random
import platform

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

torch.backends.cudnn.benchmark = True

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

init_param = 1  # int(input("use previous weights? (0 -> no, 1 -> yes): "))

n_sample = 1000

fault = nominal_params["fault"]

fault_control_index = 1

t = TicToc()

gpu_id = 0 # torch.cuda.current_device()

if platform.uname()[1] == 'realm2':
    gpu_id = 2

def main(args):
    fault = 1
    fault_control_index = args.fault_index
    traj_len = args.traj_len
    num_traj_factor = 1.99
    str_data = './data/CF_gamma_NN_weightssingle1.pth'
    str_good_data = './good_data/data/CF_gamma_NN_weightssingle1.pth'
    nominal_params["fault"] = fault
    dynamics = CrazyFlies(x=x0, goal=xg, nominal_params=nominal_params, dt=dt)
    util = Utils(n_state=n_state, m_control=m_control, dyn=dynamics, params=nominal_params, fault=fault,
                 fault_control_index=fault_control_index)
    cbf = CBF(dynamics=dynamics, n_state=n_state, m_control=m_control, fault=fault,
              fault_control_index=fault_control_index)
    gamma = Gamma(n_state=n_state, m_control=m_control, traj_len=traj_len)

    if init_param == 1:
        try:
            gamma.load_state_dict(torch.load(str_good_data))
            gamma.eval()
        except:
            print("No good data available")
            try:
                gamma.load_state_dict(torch.load(str_data))
                gamma.eval()
            except:
                print("No pre-train data available")
    
    cbf.load_state_dict(torch.load('./good_data/data/CF_cbf_NN_weightsCBF.pth'))
    cbf.eval()
    
    loss_np = 1.0
    safety_rate = 0.0

    sm, sl = dynamics.state_limits()
    
    loss_current = 0.1

    acc_ind = torch.zeros(1, m_control+1)

    for i in range(1):
        
        new_goal = dynamics.sample_safe(1)

        new_goal = new_goal.reshape(n_state, 1)

        gamma_actual_bs = torch.ones(n_sample, m_control)

        for j in range(n_sample):
            temp_var = np.mod(j, 2)
            if temp_var < 1:
                gamma_actual_bs[j, fault_control_index] = 0.0

        rand_ind = torch.randperm(n_sample)

        gamma_actual_bs = gamma_actual_bs[rand_ind, :]
        
        state = dynamics.sample_safe(n_sample) # + torch.randn(n_sample, n_state) * 2
                
        state_no_fault = state.clone()

        # state_gamma = state.clone()

        u_nominal = dynamics.u_nominal(state)
        
        t.tic()
        
        state_traj = torch.zeros(n_sample, int(num_traj_factor * traj_len), n_state)
    
        state_traj_diff = state_traj.clone()

        u_traj = torch.zeros(n_sample, int(num_traj_factor * traj_len), m_control)

        for k in range(int(traj_len * num_traj_factor)):
            
            u_nominal = dynamics.u_nominal(state, op_point=new_goal)

            fx = dynamics._f(state, params=nominal_params)
            gx = dynamics._g(state, params=nominal_params)

            u = u_nominal.clone()

            state_traj[:, k, :] = state.clone()
            
            state_traj_diff[:, k, :] = state_no_fault.clone() - state.clone()
            
            u_traj[:, k, :] = u.clone()

            # state_traj_gamma[:, k, :] = state_gamma.clone()
            gxu_no_fault = torch.matmul(gx, u.reshape(n_sample, m_control, 1))

            if k >= traj_len - 2:
                u = u * gamma_actual_bs

            # u = u * gamma_applied
            
            gxu = torch.matmul(gx, u.reshape(n_sample, m_control, 1))

            dx = fx.reshape(n_sample, n_state) + gxu.reshape(n_sample, n_state)

            dx_no_fault = fx.reshape(n_sample, n_state) + gxu_no_fault.reshape(n_sample, n_state)
            
            state_no_fault = state.clone() + dx_no_fault * dt 

            state = state.clone() + dx * dt
            
            for j2 in range(n_state):
                ind_sm = state[:, j2] > sm[j2]
                if torch.sum(ind_sm) > 0:
                    state[ind_sm, j2] = sm[j2].repeat(torch.sum(ind_sm),)
                ind_sl = state[:, j2] < sl[j2]
                if torch.sum(ind_sl) > 0:
                    state[ind_sl, j2] = sl[j2].repeat(torch.sum(ind_sl),)
            
            is_safe = int(torch.sum(util.is_safe(state))) / n_sample

            safety_rate = (i * safety_rate + is_safe) / (i + 1)
                
            if k >= traj_len - 1:
        
                gamma_NN = gamma(state_traj[:, k - traj_len + 1:k + 1, :], state_traj_diff[:, k - traj_len + 1:k + 1, :], u_traj[:, k - traj_len + 1:k + 1, :])
            
                gamma_pred = gamma_NN.reshape(n_sample, m_control).clone().detach()         
                
                for j in range(m_control):
                    index_fault = gamma_actual_bs[:, j] == 0
                    index_num = torch.sum(index_fault == True)
                    acc_ind[0, j] = 1 - torch.abs(torch.sum(gamma_actual_bs[index_fault, j] - gamma_pred[index_fault, j]) / (index_num + 1e-5))
                
                index_no_fault = torch.sum(gamma_actual_bs, dim=1) == m_control
                index_num = torch.sum(index_no_fault == True)
                acc_ind[0, -1] = torch.sum(gamma_pred[index_no_fault, :]) / (index_num + 1e-5) / m_control
                
                print('{}, {:.3f}, {:.3f}'.format(np.min([k - (traj_len - 2), traj_len]), acc_ind[0][1], acc_ind[0][-1]))
                print(torch.sum(torch.abs(gamma_pred[:, 0]-gamma_actual_bs[:, 0])) / n_sample / acc_ind[0, 0])
                print(torch.sum(torch.abs(gamma_pred[:, 1]-gamma_actual_bs[:, 1])) / n_sample / acc_ind[0, 1])
                print(torch.sum(torch.abs(gamma_pred[:, 2]-gamma_actual_bs[:, 2])) / n_sample / acc_ind[0, 2])
                print(torch.sum(torch.abs(gamma_pred[:, 3]-gamma_actual_bs[:, 3])) / n_sample / acc_ind[0, 3])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-fault_index', type=int, default=1)
    parser.add_argument('-traj_len', type=int, default=100)
    args = parser.parse_args()
    main(args)