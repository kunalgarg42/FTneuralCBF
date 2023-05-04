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
# from trainer.constraints_crazy import constraints
# from trainer.datagen import Dataset_with_Grad
# from trainer.trainer import Trainer
from trainer.utils import Utils
from trainer.NNfuncgrad_CF import CBF, Gamma_linear_LSTM, Gamma

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

use_nom = 0

gpu_id = -1

def main(args):
    fault = 1
    
    fault_control_index = args.fault_index
    
    traj_len = args.traj_len
    
    use_nom = args.use_nom

    num_traj_factor = 3
    
    # str_data = './data/CF_gamma_NN_class_linear_ALL_faults_no_res_LSTM_new.pth'
    # str_good_data = './good_data/data/CF_gamma_NN_class_linear_ALL_faults_no_res_LSTM_new.pth'
    # str_supercloud_data = './supercloud_data/CF_gamma_NN_class_linear_ALL_faults_no_res_LSTM_new.pth'
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
            # try:
            #     gamma.load_state_dict(torch.load(str_supercloud_data))
            #     # gamma.eval()
            # except:
            gamma.load_state_dict(torch.load(str_good_data))
                # gamma.eval()
        except:
            print("No good data available")
            # try:
            gamma.load_state_dict(torch.load(str_data))
                # gamma.eval()
            # except:
            #     print("No pre-train data available")
    gamma.eval()

    cbf.load_state_dict(torch.load('./data/CF_cbf_NN_weightsCBF.pth'))
    cbf.eval()

    safety_rate = 0.0

    sm, sl = dynamics.state_limits()
    
    # loss_current = 0.1

    for i in range(1000):
        
        new_goal = dynamics.sample_safe(1)

        new_goal = new_goal.reshape(n_state, 1)

        gamma_actual_bs = torch.ones(n_sample, m_control)

        for j in range(n_sample):
            temp_var = np.mod(j, 5)
            if temp_var < 1:
                gamma_actual_bs[j, temp_var] = 0.0

        rand_ind = torch.randperm(n_sample)

        gamma_actual_bs = gamma_actual_bs[rand_ind, :]
        
        state = dynamics.sample_safe(n_sample) # + torch.randn(n_sample, n_state) * 2
                
        state_no_fault = state.clone()

        u_nominal = dynamics.u_nominal(state)
        
        t.tic()
        
        state_traj = torch.zeros(n_sample, int(num_traj_factor * traj_len), n_state)
    
        state_traj_diff = state_traj.clone()

        u_traj = torch.zeros(n_sample, int(num_traj_factor * traj_len), m_control)

        acc_ind_temp = torch.zeros(1, 2 * m_control)

        for k in range(int(traj_len * num_traj_factor)):
            
            u_nominal = dynamics.u_nominal(state, op_point=new_goal)

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

            if k >= traj_len - 1:
                u = u * gamma_actual_bs
            
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
                gamma_data = gamma(state_traj[:, k-traj_len + 1:k + 1, :], state_traj_diff[:, k-traj_len + 1:k+1, :], u_traj[:, k-traj_len + 1:k + 1, :])

                gamma_data = gamma_data.detach()

                for j in range(m_control):
                        
                    index_fault = (gamma_actual_bs[:, j]- 0.5) < 0
                    
                    index_num = torch.sum(index_fault.float())

                    acc_ind_temp[0, j] = torch.sum((gamma_data[index_fault, j] < 0).float()) / (index_num + 1e-5)
                    
                    index_no_fault = gamma_actual_bs[:, j] - 0.5 > 0

                    index_num = torch.sum(index_no_fault.float())

                    acc_ind_temp[0, m_control + j] = torch.sum((gamma_data[index_no_fault, j]> 0).float()) / (index_num + 1e-5)
                
                print(acc_ind_temp[0])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-fault_index', type=int, default=1)
    parser.add_argument('-use_nom', type=int, default=1)
    parser.add_argument('-traj_len', type=int, default=100)
    args = parser.parse_args()
    main(args)