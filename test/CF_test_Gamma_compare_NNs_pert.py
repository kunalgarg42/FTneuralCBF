import os
import sys
import torch
import numpy as np
import argparse
import tqdm
import matplotlib.pyplot as plt

sys.path.insert(1, os.path.abspath('..'))
sys.path.insert(1, os.path.abspath('.'))

from pytictoc import TicToc
from dynamics.Crazyflie import CrazyFlies
from trainer import config
from trainer.utils import Utils
from trainer.NNfuncgrad_CF import Gamma, CBF, Gamma_linear_LSTM, Gamma_linear_conv, Gamma_linear_deep_nonconv, Gamma_linear_nonconv, Gamma_linear_LSTM_old, Gamma_linear_LSTM_small


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

nominal_params = config.CRAZYFLIE_PARAMS_PERT

fault = nominal_params["fault"]

use_good = 1

traj_len = 100

Eval_steps = 199

fault = nominal_params["fault"]

t = TicToc()

gpu_id = 3

gamma_type = 'LSTM old'

def main(args):

    nominal_params = config.CRAZYFLIE_PARAMS_PERT

    dynamics = CrazyFlies(x=x0, goal=xg, nominal_params=nominal_params, dt=dt)

    use_nom = args.use_nom

    if use_nom == 1:
        n_sample = 1000
        # sys.stdout = open('./log_files/log_gamma_train_LQR_' + gamma_type + '.txt', 'w')
    else:
        n_sample = 1000
        # sys.stdout = open('./log_files/log_gamma_train_CBF_' + gamma_type + '.txt', 'w')
    
    if use_nom == 1:
        nsample_factor = 1
    else:
        nsample_factor = 1

    n_sample_iter = int(n_sample / nsample_factor)

    acc_final = torch.zeros(4, 2, traj_len)

    state0 = dynamics.sample_safe(n_sample_iter)

    new_goal = dynamics.sample_safe(1)
    
    for gamma_iter in range(4):
        if gamma_iter > 2:
            nominal_params = config.CRAZYFLIE_PARAMS_PERT
        else:
            nominal_params = config.CRAZYFLIE_PARAMS

        if np.mod(gamma_iter, 2) == 0:
            gamma_type = 'LSTM'
            fault_control_index = 0
        else:
            gamma_type = 'old'
            fault_control_index = 1
        
        dynamics = CrazyFlies(x=x0, goal=xg, nominal_params=nominal_params, dt=dt)

        sm, sl = dynamics.state_limits()
    
        util = Utils(n_state=n_state, m_control=m_control, dyn=dynamics, params=nominal_params, fault=fault,
                    fault_control_index=fault_control_index)
        
        cbf = CBF(dynamics=dynamics, n_state=n_state, m_control=m_control, fault=fault,
                fault_control_index=fault_control_index)

        cbf.load_state_dict(torch.load('./supercloud_data/CF_cbf_NN_weightsCBF_with_u.pth'))
        cbf.eval()
        # if gamma_iter == 0:
        #     gamma_type = 'LSTM'
        # elif gamma_iter == 1:
        #     gamma_type = 'deep'
        # elif gamma_iter == 2:
        #     gamma_type = 'linear conv'
        # else:
        #     gamma_type = 'old'

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
        else:
            str_data = './data/CF_gamma_NN_weightssingle1.pth'
            str_good_data = './good_data/data/CF_gamma_NN_weightssingle1.pth'
        
        if gamma_type == 'linear nonconv':
            gamma = Gamma_linear_nonconv(n_state=n_state, m_control=m_control, traj_len=traj_len)
        elif gamma_type == 'deep':
            gamma = Gamma_linear_deep_nonconv(n_state=n_state, m_control=m_control, traj_len=traj_len)
        elif gamma_type == 'LSTM':
            gamma = Gamma_linear_LSTM(n_state=n_state, m_control=m_control, traj_len=traj_len)
        elif gamma_type == 'LSTM old':
            gamma = Gamma_linear_LSTM_old(n_state=n_state, m_control=m_control, traj_len=traj_len)
        elif gamma_type == 'linear conv':
            gamma = Gamma_linear_conv(n_state=n_state, m_control=m_control, traj_len=traj_len)
        elif gamma_type == 'LSTM small':
            gamma = Gamma_linear_LSTM_small(n_state=n_state, m_control=m_control, traj_len=traj_len)
        else:
            gamma = Gamma(n_state=n_state, m_control=m_control, traj_len=traj_len)

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
            
        acc0 = torch.zeros(traj_len + 1, 1)
        acc1 = torch.zeros(traj_len + 1, 1)

        gamma_actual_bs = torch.ones(n_sample_iter, m_control)

        if gamma_type != 'old':
            for j in range(n_sample_iter):
                temp_var = np.mod(j, 5)
                if temp_var < 4:
                    gamma_actual_bs[j, temp_var] = 0.0
        else:
            for j in range(n_sample):
                temp_var = np.mod(j, 2)
                if temp_var < 1:
                    gamma_actual_bs[j, fault_control_index] = 0.0
        
        rand_ind = torch.randperm(n_sample_iter)

        gamma_actual_bs = gamma_actual_bs[rand_ind, :]

        
        state_traj = torch.zeros(n_sample_iter, Eval_steps, n_state)    

        state_traj_diff = torch.zeros(n_sample_iter, Eval_steps, n_state)

        u_traj = torch.zeros(n_sample_iter, Eval_steps, m_control)
        
        state = state0.clone()

        state_no_fault = state.clone()

        for k in range(n_state):
            if k > 5:
                state[:, k] = torch.clamp(state[:, k], sm[k] / 10, sl[k] / 10)
        
        u_nominal = dynamics.u_nominal(state)
        
        t.tic()

        # print('length of failure, acc fail , acc no fail')

        new_goal = new_goal.reshape(n_state, 1)

        for k in tqdm.trange(Eval_steps):

            u_nominal = dynamics.u_nominal(state, op_point=new_goal)

            fx = dynamics._f(state, params=nominal_params)
            gx = dynamics._g(state, params=nominal_params)

            if use_nom == 0:
                h, grad_h = cbf.V_with_jacobian(state.reshape(n_sample_iter, n_state, 1))
                u = util.fault_controller(u_nominal, fx, gx, h, grad_h)
            else:
                u = u_nominal.clone()

            state_traj[:, k, :] = state.clone()

            state_traj_diff[:, k, :] = state_no_fault.clone() - state.clone()
                    
            u_traj[:, k, :] = u.clone()

            gxu_no_fault = torch.matmul(gx, u.reshape(n_sample, m_control, 1))
            
            if k >= traj_len - 1:
                u = u * gamma_actual_bs
            
            gxu = torch.matmul(gx, u.reshape(n_sample_iter, m_control, 1))

            dx = fx.reshape(n_sample_iter, n_state) + gxu.reshape(n_sample_iter, n_state)

            dx_no_fault = fx.reshape(n_sample, n_state) + gxu_no_fault.reshape(n_sample, n_state)

            state = state.clone() + dx * dt

            state_no_fault = state.clone() + dx_no_fault * dt 
        
            for j2 in range(n_state):
                ind_sm = state[:, j2] > sm[j2]
                if torch.sum(ind_sm) > 0:
                    state[ind_sm, j2] = sm[j2].repeat(torch.sum(ind_sm),)
                ind_sl = state[:, j2] < sl[j2]
                if torch.sum(ind_sl) > 0:
                    state[ind_sl, j2] = sl[j2].repeat(torch.sum(ind_sl),)

            if k >= traj_len - 1:
                
                if gamma_type != 'old':
                    gamma_NN = gamma(state_traj[:, k - traj_len + 1:k + 1, :], u_traj[:, k - traj_len + 1:k + 1, :])
                else:
                    gamma_NN = gamma(state_traj[:, k - traj_len + 1:k + 1, :], state_traj_diff[:, k - traj_len + 1:k + 1, :], u_traj[:, k - traj_len + 1:k + 1, :])

                gamma_pred = gamma_NN.reshape(n_sample_iter, m_control).clone().detach()

                if gamma_type != 'old':
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
                    # acc_final[gamma_iter, 0, k-traj_len+1] = torch.min(acc_ind[0, 0:m_control])
                    # acc_final[gamma_iter, 0, k-traj_len+1] = acc_ind[0, fault_control_index]
                    acc_final[gamma_iter, 0, k-traj_len+1] = torch.sum(acc_ind[0, 0:m_control]) / m_control
                    acc_final[gamma_iter, 1, k-traj_len+1] = torch.min(acc_ind[0, m_control:])
                else:
                    acc_ind = torch.zeros(1, m_control+1)            
            
                    for j in range(m_control):
                        index_fault = gamma_actual_bs[:, j]==0
                        index_num = torch.sum(index_fault.float())
                        acc_ind[0, j] = 1 - torch.abs(torch.sum(gamma_actual_bs[index_fault, j] - gamma_pred[index_fault, j]) / (index_num + 1e-5))
                    
                    index_no_fault = torch.sum(gamma_actual_bs, dim=1) == m_control
                    
                    index_num = torch.sum(index_no_fault.float())
                    
                    acc_ind[0, -1] = torch.sum(gamma_pred[index_no_fault, :]) / (index_num + 1e-5) / m_control
                
                    acc_final[gamma_iter, 0, k-traj_len+1] = acc_ind[0, fault_control_index]
                    acc_final[gamma_iter, 1, k-traj_len+1] = acc_ind[0, -1]

    fig = plt.figure(figsize=(12, 6))
    ax = fig.subplots(1, 1)
    
    plot_name = './plots/' + 'CF_gamma_compare_all_NN_pert' + '.png'

    # ax = axes[0]

    step = np.arange(0, traj_len, 1)

    markers_on = np.arange(0, step[-1], 10)

    colors = ['b', 'm', 'c', 'r', 'g', 'y', 'k', 'w']

    for gamma_iter in range(4):
        if gamma_iter == 0:
            gamma_type = 'Model-free (nominal)'
        elif gamma_iter == 1:
            gamma_type = 'Model-based (nominal)'
        elif gamma_iter == 2:
            gamma_type = 'Model-free (perturbed)'
        else:
            gamma_type = 'Model-based (perturbed)'

        acc_fail = acc_final[gamma_iter, 0, :]
        acc_no_fail = acc_final[gamma_iter, 1, :]

        ax.plot(step, acc_fail, color = colors[gamma_iter], linestyle='-', label = 'Failure: ' + gamma_type, marker="o", markevery=markers_on,markersize=10)
        ax.plot(step, acc_no_fail, color = colors[gamma_iter], linestyle='--', label = 'No Failure: ' + gamma_type, marker="^", markevery=markers_on,markersize=10)

    plt.xlabel('Length of trajectory with failed actuator', fontsize = 20)
    plt.ylabel('Accuracy', fontsize = 20)
    
    # plt.title('Failure Test Accuracy', fontsize = 20)
    plt.legend(fontsize=15, ncol =1, loc='upper left', bbox_to_anchor=(1.0, 0.8))
    plt.tight_layout()
    ax.set_xlim(step[0], step[-1])
    ax.set_ylim(0.5, 1.05)
    ax.tick_params(axis = "x", labelsize = 15)
    ax.tick_params(axis = "y", labelsize = 15)

    plt.savefig(plot_name)

    print("saved file:", plot_name)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-fault_index', type=int, default=1)
    parser.add_argument('-use_nom', type=int, default=1)
    args = parser.parse_args()
    main(args)
