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
from trainer.NNfuncgrad_CF import CBF, Gamma_linear_LSTM, Gamma_linear, Gamma_linear_deep_nonconv, Gamma_linear_nonconv, Gamma_linear_LSTM_old, Gamma_linear_LSTM_small


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

traj_len = 100

Eval_steps = 199

fault = nominal_params["fault"]

t = TicToc()

gpu_id = 3

gamma_type = 'LSTM old'

def main(args):
    fault_control_index = args.fault_index
    use_nom = args.use_nom

    if use_nom == 1:
        n_sample = 10000
        sys.stdout = open('./log_files/log_gamma_train_LQR_' + gamma_type + '.txt', 'w')
    else:
        n_sample = 1000
        sys.stdout = open('./log_files/log_gamma_train_CBF_' + gamma_type + '.txt', 'w')

    if gamma_type == 'LSTM':
        str_data = './data/CF_gamma_NN_class_linear_ALL_faults_no_res_LSTM_new.pth'
        str_good_data = './good_data/data/CF_gamma_NN_class_linear_ALL_faults_no_res_LSTM_new.pth'
        str_supercloud_data = './supercloud_data/CF_gamma_NN_class_linear_ALL_faults_no_res_LSTM_new.pth'
    elif gamma_type == 'LSTM old':
        str_data = './data/CF_gamma_NN_class_linear_ALL_faults_no_res_LSTM.pth'
        str_good_data = './good_data/data/CF_gamma_NN_class_linear_ALL_faults_no_res_LSTM.pth'
        str_supercloud_data = './supercloud_data/CF_gamma_NN_class_linear_ALL_faults_no_res_LSTM.pth'
    elif gamma_type == 'LSTM small':
        str_data = './data/CF_gamma_NN_class_linear_ALL_faults_no_res_LSTM_small.pth'
        str_good_data = './good_data/data/CF_gamma_NN_class_linear_ALL_faults_no_res_LSTM_small.pth'
        str_supercloud_data = './supercloud_data/CF_gamma_NN_class_linear_ALL_faults_no_res_LSTM_small.pth'
    elif gamma_type == 'linear nonconv':
        str_data = './data/CF_gamma_NN_class_linear_ALL_faults_no_res.pth'
        str_good_data = './good_data/data/CF_gamma_NN_class_linear_ALL_faults_no_res.pth'
        str_supercloud_data = './supercloud_data/CF_gamma_NN_class_linear_ALL_faults_no_res.pth'
    elif gamma_type == 'deep':
        str_data = './data/CF_gamma_NN_class_linear_ALL_faults_no_res_non_conv_deep.pth'
        str_good_data = './good_data/data/CF_gamma_NN_class_linear_ALL_faults_no_res_non_conv_deep.pth'
        str_supercloud_data = './supercloud_data/CF_gamma_NN_class_linear_ALL_faults_no_res_non_conv_deep.pth'
    elif gamma_type == 'linear conv':
        str_data = './data/CF_gamma_NN_class_linear_ALL_faults.pth'
        str_good_data = './good_data/data/CF_gamma_NN_class_linear_ALL_faults.pth'
        str_supercloud_data = './supercloud_data/CF_gamma_NN_class_linear_ALL_faults.pth'
    
    dynamics = CrazyFlies(x=x0, goal=xg, nominal_params=nominal_params, dt=dt)
    util = Utils(n_state=n_state, m_control=m_control, dyn=dynamics, params=nominal_params, fault=fault,
                 fault_control_index=fault_control_index)
    cbf = CBF(dynamics=dynamics, n_state=n_state, m_control=m_control, fault=fault,
              fault_control_index=fault_control_index)
    
    if gamma_type == 'linear nonconv':
        gamma = Gamma_linear_nonconv(n_state=n_state, m_control=m_control, traj_len=traj_len)
    elif gamma_type == 'deep':
        gamma = Gamma_linear_deep_nonconv(n_state=n_state, m_control=m_control, traj_len=traj_len)
    elif gamma_type == 'LSTM':
        gamma = Gamma_linear_LSTM(n_state=n_state, m_control=m_control, traj_len=traj_len)
    elif gamma_type == 'LSTM old':
        gamma = Gamma_linear_LSTM_old(n_state=n_state, m_control=m_control, traj_len=traj_len)
    elif gamma_type == 'linear conv':
        gamma = Gamma_linear(n_state=n_state, m_control=m_control, traj_len=traj_len)
    elif gamma_type == 'LSTM small':
        gamma = Gamma_linear_LSTM_small(n_state=n_state, m_control=m_control, traj_len=traj_len)

    if use_good == 1:
        try:
            if gpu_id == -1:
                gamma.load_state_dict(torch.load(str_supercloud_data, map_location=torch.device('cpu')))
            else:
                gamma.load_state_dict(torch.load(str_supercloud_data))
        except:
            try:
                if gpu_id == -1:
                    gamma.load_state_dict(torch.load(str_good_data, map_location=torch.device('cpu')))
                else:
                    gamma.load_state_dict(torch.load(str_good_data))
            except:
                if gpu_id == -1:
                    gamma.load_state_dict(torch.load(str_data, map_location=torch.device('cpu')))
                else:
                    gamma.load_state_dict(torch.load(str_data))
                print("No good data available")
    else:
        gamma.load_state_dict(torch.load(str_data))
    
    gamma.eval()
        
    cbf.load_state_dict(torch.load('./supercloud_data/CF_cbf_NN_weightsCBF_with_u.pth'))
    cbf.eval()

    sm, sl = dynamics.state_limits()
    
    if use_nom == 1:
        nsample_factor = 1
    else:
        nsample_factor = 1

    n_sample_iter = int(n_sample / nsample_factor)

    acc0 = torch.zeros(traj_len + 1, 1)
    acc1 = torch.zeros(traj_len + 1, 1)

    gamma_actual_bs = torch.ones(n_sample_iter, m_control)

    for j in range(n_sample_iter):
        temp_var = np.mod(j, 5)
        if temp_var < 4:
            gamma_actual_bs[j, temp_var] = 0.0

    for i in range(nsample_factor):
        # if i < m_control:
        #     gamma_actual_bs[i, :] = 0.0

        rand_ind = torch.randperm(n_sample_iter)

        gamma_actual_bs = gamma_actual_bs[rand_ind, :]

        state0 = dynamics.sample_safe(n_sample_iter)

        state_traj = torch.zeros(n_sample_iter, Eval_steps, n_state)    

        u_traj = torch.zeros(n_sample_iter, Eval_steps, m_control)
        
        state = state0.clone()

        for k in range(n_state):
            if k > 5:
                state[:, k] = torch.clamp(state[:, k], sm[k] / 10, sl[k] / 10)
        
        u_nominal = dynamics.u_nominal(state)
        
        t.tic()

        # print('length of failure, acc fail , acc no fail')

        new_goal = dynamics.sample_safe(1)

        new_goal = new_goal.reshape(n_state, 1)

        for k in range(Eval_steps):

            u_nominal = dynamics.u_nominal(state, op_point=new_goal)

            fx = dynamics._f(state, params=nominal_params)
            gx = dynamics._g(state, params=nominal_params)

            if use_nom == 0:
                h, grad_h = cbf.V_with_jacobian(state.reshape(n_sample_iter, n_state, 1))
                u = util.fault_controller(u_nominal, fx, gx, h, grad_h)
            else:
                u = u_nominal.clone()

            state_traj[:, k, :] = state.clone()
                    
            u_traj[:, k, :] = u.clone()
            
            if k >= traj_len - 1:
                u = u * gamma_actual_bs
            
            gxu = torch.matmul(gx, u.reshape(n_sample_iter, m_control, 1))

            dx = fx.reshape(n_sample_iter, n_state) + gxu.reshape(n_sample_iter, n_state)

            state = state.clone() + dx * dt
        
            for j2 in range(n_state):
                ind_sm = state[:, j2] > sm[j2]
                if torch.sum(ind_sm) > 0:
                    state[ind_sm, j2] = sm[j2].repeat(torch.sum(ind_sm),)
                ind_sl = state[:, j2] < sl[j2]
                if torch.sum(ind_sl) > 0:
                    state[ind_sl, j2] = sl[j2].repeat(torch.sum(ind_sl),)

            if k >= traj_len - 1:
                if gamma_type == 'linear conv':
                    gamma_NN = gamma(state_traj[:, k - traj_len + 1:k + 1, :], 0 * state_traj[:, k - traj_len + 1:k + 1, :], u_traj[:, k - traj_len + 1:k + 1, :])
                else:    
                    gamma_NN = gamma(state_traj[:, k - traj_len + 1:k + 1, :], u_traj[:, k - traj_len + 1:k + 1, :])

                gamma_pred = gamma_NN.reshape(n_sample_iter, m_control).clone().detach()

                acc_ind = torch.zeros(1, m_control * 2)

                for j in range(m_control):
                    
                    index_fault = gamma_actual_bs[:, j] < 0.5

                    index_num = torch.sum(index_fault.float())

                    if index_num > 0:
                        acc_ind[0, j] =  torch.sum((gamma_pred[index_fault, j] < 0).float()) / (index_num + 1e-5)
                    else:
                        acc_ind[0, j] = 1
                    
                    index_no_fault = gamma_actual_bs[:, j] > 0.5
                    
                    index_num = torch.sum(index_no_fault.float())

                    if index_num > 0:
                        acc_ind[0, j + m_control] =  torch.sum((gamma_pred[index_no_fault, j] > 0).float()) / (index_num + 1e-5)
                    else:
                        acc_ind[0, j + m_control] = 1
                
                np.set_printoptions(precision=3, suppress=True)

                text_print = '{}, {}'.format(np.max([np.min([k - (traj_len - 2), traj_len]), 0]), acc_ind[0].numpy())
                
                sys.stdout.write(text_print + '\n')
                sys.stdout.flush()

    #             acc0[np.max([np.min([k - (traj_len - 2), traj_len]), 0])] += torch.min(acc_ind[0, 0:m_control])
    #             acc1[np.max([np.min([k - (traj_len - 2), traj_len]), 0])] += torch.min(acc_ind[0, m_control:2 * m_control])
    
    # acc0 = acc0 / nsample_factor
    # acc1 = acc1 / nsample_factor
    
    # np.set_printoptions(precision=3, suppress=True)

    # for i in range(traj_len + 1):
    #     text_print = '{}, {}, {}'.format(i, acc0[i].numpy(), acc1[i].numpy())

    #     sys.stdout.write(text_print + '\n')
    #     sys.stdout.flush()

                

    sys.stdout.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-fault_index', type=int, default=1)
    parser.add_argument('-use_nom', type=int, default=1)
    args = parser.parse_args()
    main(args)
