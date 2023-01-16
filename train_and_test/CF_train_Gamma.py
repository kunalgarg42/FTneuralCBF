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

n_sample = 250

# n_sample_data = 500

fault = nominal_params["fault"]

fault_control_index = 0

t = TicToc()

gpu_id = 0 # torch.cuda.current_device()

if platform.uname()[1] == 'realm2':
    gpu_id = 2

def main(args):
    fault = 1
    fault_control_index = args.fault_index
    traj_len = args.traj_len
    num_traj_factor = int(1.6 - traj_len / 1000)
    data_index = int((traj_len - 100) / 100)
    str_data = './data/CF_gamma_NN_weights{}.pth'.format(data_index)
    str_good_data = './good_data/data/CF_gamma_NN_weights{}.pth'.format(data_index)
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

    dataset = Dataset_with_Grad(n_state=n_state, m_control=m_control, train_u=0, buffer_size=n_sample*500, traj_len=traj_len)
    trainer = Trainer(cbf, dataset, gamma=gamma, n_state=n_state, m_control=m_control, j_const=2, dyn=dynamics,
                      dt=dt, action_loss_weight=0.001, params=nominal_params,
                      fault=fault, gpu_id=gpu_id, num_traj=n_sample, traj_len=traj_len,
                      fault_control_index=fault_control_index)
    loss_np = 1.0
    safety_rate = 0.0

    sm, sl = dynamics.state_limits()
    
    loss_current = 100.0
    # C1 = np.eye(n_state)
    # C2 = np.array([0]*n_state*m_control).reshape(n_state, m_control)
    # C_EKF = np.hstack((C1, C2))
    
    # state_EKF = torch.ones(n_state + m_control, 1)
    
    # P = np.eye(n_state + m_control)

    for i in range(1000):
        
        new_goal = torch.randn(n_state, 1)

        gamma_actual_bs = torch.ones(n_sample, m_control)
        # gamma_fault_rand = torch.rand() / 4
        fault_index_extra = random.randint(0, m_control-1)
        for j in range(n_sample):
            fault_control_index = np.mod(j, 6)
            if fault_control_index < m_control:
                gamma_actual_bs[j, fault_control_index] = 0.0
            if fault_control_index == 5:
                gamma_actual_bs[j, fault_index_extra] = 0.0
            # else:
                # gamma_actual_bs[j, fault_control_index]
        # fault_control_index = int(np.mod(i, 8) / 2)
        # gamma_actual_bs[:, fault_control_index] = torch.ones(n_sample,) * 0.2
        rand_ind = torch.randperm(n_sample)
        gamma_actual_bs = gamma_actual_bs[rand_ind, :]

        # dataset.add_data(torch.tensor([]).reshape(0, traj_len, n_state), torch.tensor([]).reshape(0, traj_len, n_state), torch.tensor([]).reshape(0, traj_len, m_control), gamma_actual_bs)
        
        state = dynamics.sample_safe(n_sample)
                
        state_no_fault = state.clone()

        # state_gamma = state.clone()

        u_nominal = dynamics.u_nominal(state)
        
        t.tic()
        
        gamm_pred = torch.ones(n_sample, m_control)

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
            
            u = u * gamma_actual_bs * gamm_pred
            
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
            
            is_safe = int(torch.sum(util.is_safe(state))) / n_sample

            safety_rate = (i * safety_rate + is_safe) / (i + 1)
        
        # print(state_EKF[-m_control:])
        
        # dataset.add_data(state_traj.reshape(n_sample * traj_len, n_state), 1 * state_traj_diff.reshape(n_sample * traj_len, n_state), u_traj.reshape(n_sample * traj_len, m_control), torch.tensor([]).reshape(0, m_control))
            if k >= traj_len - 1:
                dataset.add_data(state_traj[:, k-traj_len + 1:k + 1, :], state_traj_diff[:, k-traj_len + 1:k + 1, :], u_traj[:, k-traj_len + 1:k + 1, :], gamma_actual_bs)
                    # state_traj = state_traj[:, -traj_len:, :]
                    # state_traj_diff = state_traj_diff[:, -traj_len:, :]
                    # u_traj = u_traj[:, -traj_len:, :]

        loss_np, acc_np, acc_ind = trainer.train_gamma()

        time_iter = t.tocvalue()
        print(
            'step, {}, loss, {:.3f}, acc, {:.3f}, acc ind, {}, safety rate, {:.3f}, time, {:.3f} '.format(
                i, loss_np, acc_np, acc_ind, safety_rate, time_iter))

        if loss_np <= loss_current and i > 5:
            loss_current = loss_np.copy()
            torch.save(gamma.state_dict(), str_data)

            if loss_np < 0.01 or acc_np > 0.96:
                torch.save(gamma.state_dict(), str_good_data)
        
        if loss_np < 0.001 and i > 250:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-fault_index', type=int, default=0)
    parser.add_argument('-traj_len', type=int, default=100)
    args = parser.parse_args()
    main(args)
