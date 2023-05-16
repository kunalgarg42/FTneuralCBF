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
from trainer.datagen import Dataset_with_Grad
from trainer.trainer import Trainer
from trainer.utils import Utils
from trainer.NNfuncgrad_CF import CBF, Gamma_linear_LSTM, Gamma_linear_conv, Gamma_linear_deep_nonconv, Gamma_linear_nonconv, Gamma_linear_LSTM_old, Gamma_linear_LSTM_small

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
    gpu_id = 1

def main(args):
    fault = 1
    fault_control_index = args.fault_index
    traj_len = args.traj_len
    gamma_type = args.gamma_type
    # gamma_type = 'LSTM small'
    num_traj_factor = 2

    if gamma_type == 'LSTM':
        str_data = './data/CF_gamma_NN_class_linear_ALL_faults_no_res_LSTM_new.pth'
        str_good_data = './good_data/data/CF_gamma_NN_class_linear_ALL_faults_no_res_LSTM_new.pth'
    elif gamma_type == 'LSTM old':
        str_data = './data/CF_gamma_NN_class_linear_ALL_faults_no_res_LSTM.pth'
        str_good_data = './good_data/data/CF_gamma_NN_class_linear_ALL_faults_no_res_LSTM.pth'
    elif gamma_type == 'LSTM small':
        str_data = './data/CF_gamma_NN_class_linear_ALL_faults_no_res_LSTM_small.pth'
        str_good_data = './good_data/data/CF_gamma_NN_class_linear_ALL_faults_no_res_LSTM_small.pth'
    elif gamma_type == 'linear nonconv':
        str_data = './data/CF_gamma_NN_class_linear_ALL_faults_no_res.pth'
        str_good_data = './good_data/data/CF_gamma_NN_class_linear_ALL_faults_no_res.pth'
    elif gamma_type == 'deep':
        str_data = './data/CF_gamma_NN_class_linear_ALL_faults_no_res_non_conv_deep.pth'
        str_good_data = './good_data/data/CF_gamma_NN_class_linear_ALL_faults_no_res_non_conv_deep.pth'
    elif gamma_type == 'linear conv':
        str_data = './data/CF_gamma_NN_class_linear_ALL_faults.pth'
        str_good_data = './good_data/data/CF_gamma_NN_class_linear_ALL_faults.pth'

    nominal_params["fault"] = fault
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
    elif gamma_type == 'LSTM small':
        gamma = Gamma_linear_LSTM_small(n_state=n_state, m_control=m_control, traj_len=traj_len)
    elif gamma_type == 'LSTM old':
        gamma = Gamma_linear_LSTM_old(n_state=n_state, m_control=m_control, traj_len=traj_len)
    elif gamma_type == 'linear conv':
        gamma = Gamma_linear_conv(n_state=n_state, m_control=m_control, traj_len=traj_len)

    if init_param == 1:
        try:
            gamma.load_state_dict(torch.load(str_good_data))
            gamma.eval()
            if gamma_type == 'LSTM' or gamma_type == 'LSTM old' or gamma_type == 'LSTM small':
                gamma.train()
        except:
            print("No good data available")
            try:
                gamma.load_state_dict(torch.load(str_data))
                gamma.eval()
                if gamma_type == 'LSTM' or gamma_type == 'LSTM old' or gamma_type == 'LSTM small':
                    gamma.train()
            except:
                print("No pre-train data available")
    
    cbf.load_state_dict(torch.load('./data/CF_cbf_NN_weightsCBF.pth'))
    cbf.eval()

    dataset = Dataset_with_Grad(n_state=n_state, m_control=m_control, train_u=0, buffer_size=n_sample*300, traj_len=traj_len)
    trainer = Trainer(cbf, dataset, gamma=gamma, n_state=n_state, m_control=m_control, j_const=2, dyn=dynamics,
                      dt=dt, action_loss_weight=0.001, params=nominal_params,
                      fault=fault, gpu_id=gpu_id, num_traj=n_sample, traj_len=traj_len,
                      fault_control_index=fault_control_index)
    loss_np = 1.0
    safety_rate = 0.0

    sm, sl = dynamics.state_limits()
    
    loss_current = 0.1

    for i in range(1000):
        
        new_goal = dynamics.sample_safe(1)

        new_goal = new_goal.reshape(n_state, 1)

        gamma_actual_bs = torch.ones(n_sample, m_control)

        for j in range(n_sample):
            temp_var = np.mod(j, 6)
            if temp_var < 4:
                gamma_actual_bs[j, temp_var] = 0.0

        rand_ind = torch.randperm(n_sample)

        gamma_actual_bs = gamma_actual_bs[rand_ind, :]
        
        state = dynamics.sample_safe(n_sample) # + torch.randn(n_sample, n_state) * 2

        for k in range(n_state):
            if k > 5:
                state[:, k] = torch.clamp(state[:, k], sm[k] / 10, sl[k] / 10)
                
        state_no_fault = state.clone()

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
                
            if k >= traj_len -1:
                if k == traj_len - 1:
                    dataset.add_data(state_traj[:, k-traj_len + 1:k + 1, :], 0 * state_traj_diff[:, k-traj_len + 1:k + 1, :], u_traj[:, k-traj_len + 1:k + 1, :], 0.5 * torch.ones(n_sample, m_control))
                else:
                    dataset.add_data(state_traj[:, k-traj_len + 1:k + 1, :], 0 * state_traj_diff[:, k-traj_len + 1:k + 1, :], u_traj[:, k-traj_len + 1:k + 1, :], (gamma_actual_bs - 0.5))
        
        loss_np, acc_np = trainer.train_gamma()

        time_iter = t.tocvalue()
        print(
            'step, {}, loss, {:.3f}, acc, {}, safety rate, {:.3f}, time, {:.3f} '.format(
                i, loss_np, acc_np, safety_rate, time_iter))

        if (loss_np <= loss_current or np.sum(acc_np) / int(acc_np.size) > 0.96) and i > 5:
            loss_current = loss_np.copy()
            torch.save(gamma.state_dict(), str_data)

            if loss_np <= 0.01 or np.sum(acc_np) / int(acc_np.size) > 0.97:
                torch.save(gamma.state_dict(), str_good_data)
        
            if loss_np <= 0.001 and i > 250:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-fault_index', type=int, default=1)
    parser.add_argument('-traj_len', type=int, default=100)
    parser.add_argument('-gamma_type', type=str, default='linear')
    args = parser.parse_args()
    main(args)