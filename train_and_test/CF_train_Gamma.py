import os
import sys
import torch
import numpy as np
import argparse

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
                    3.5,
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
print(fault)

init_add = 1  # int(input("init data add? (0 -> no, 1 -> yes): "))
print(init_add)

init_param = 1  # int(input("use previous weights? (0 -> no, 1 -> yes): "))
print(init_param)

train_u = 0  # int(input("Train only CBF (0) or both CBF and u (1): "))
print(train_u)

n_sample = 5000

n_sample_data = 1000

traj_len = 100

fault = nominal_params["fault"]

fault_control_index = 0

str_data = './data/CF_gamma_NN_weights{}.pth'.format(fault_control_index)
str_good_data = './good_data/data/CF_gamma_NN_weights{}.pth'.format(fault_control_index)

t = TicToc()

gpu_id = 0

def main(args):
    fault = 1
    fault_control_index = args.fault_index
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

    dataset = Dataset_with_Grad(n_state=n_state, m_control=m_control, train_u=train_u, buffer_size=n_sample_data*traj_len*50, traj_len=traj_len)
    trainer = Trainer(cbf, dataset, gamma=gamma, n_state=n_state, m_control=m_control, j_const=2, dyn=dynamics,
                      dt=dt, action_loss_weight=0.001, params=nominal_params,
                      fault=fault, gpu_id=gpu_id, num_traj=n_sample, traj_len=traj_len,
                      fault_control_index=fault_control_index)
    loss_np = 1.0
    safety_rate = 0.0

    sm, sl = dynamics.state_limits()
    safe_m, safe_l = dynamics.safe_limits(sm, sl, fault)
    
    loss_current = 100.0
    # C1 = np.eye(n_state)
    # C2 = np.array([0]*n_state*m_control).reshape(n_state, m_control)
    # C_EKF = np.hstack((C1, C2))
    
    # state_EKF = torch.ones(n_state + m_control, 1)
    
    # P = np.eye(n_state + m_control)

    for i in range(1000):
        gamma_actual_bs = torch.ones(n_sample, m_control)
        # gamma_fault_rand = torch.rand() / 4

        for j in range(n_sample):
            fault_control_index = np.mod(j, 5)
            if fault_control_index < 4:
                gamma_actual_bs[j, fault_control_index] = 0.1
            # else:
                # gamma_actual_bs[j, fault_control_index]
        # fault_control_index = int(np.mod(i, 8) / 2)
        # gamma_actual_bs[:, fault_control_index] = torch.ones(n_sample,) * 0.2

        dataset.add_data(torch.tensor([]).reshape(0, n_state), torch.tensor([]).reshape(0, n_state), torch.tensor([]).reshape(0, m_control), gamma_actual_bs)

        state0 = util.x_samples(safe_m, safe_l, n_sample)
        
        state_traj = torch.zeros(n_sample, traj_len, n_state)
        
        state_traj_diff = state_traj.clone()

        u_traj = torch.zeros(n_sample, traj_len, m_control)
        
        state = state0.clone()
        
        state_no_fault = state0.clone()

        # state_gamma = state.clone()

        u_nominal = dynamics.u_nominal(state)
        
        t.tic()

        for k in range(traj_len):
            
            u_nominal = dynamics.u_nominal(state)

            fx = dynamics._f(state, params=nominal_params)
            gx = dynamics._g(state, params=nominal_params)

            # fx_gamma = dynamics._f(state_EKF[0:n_state].reshape(1, n_state), params=nominal_params)
            # gx_gamma = dynamics._g(state_EKF[0:n_state].reshape(1, n_state), params=nominal_params)

            # u_nominal_gamma = dynamics.u_nominal(state_gamma)
            # A_mat = dynamics.compute_AB_matrices(state_EKF[0:n_state].reshape(1, n_state), u_nominal[0, :].reshape(1, m_control) * state_EKF[-m_control:].reshape(1, m_control))
            
            # gxu0 = gx[0, :, 0] * u_nominal[0, 0] * state_EKF[-m_control]
            # gxu1 = gx[0, :, 1] * u_nominal[0, 1] * state_EKF[-m_control+1]
            # gxu2 = gx[0, :, 2] * u_nominal[0, 2] * state_EKF[-m_control+2]
            # gxu3 = gx[0, :, 3] * u_nominal[0, 3] * state_EKF[-m_control+3]

            # A_mat = np.hstack((A_mat, np.array(gxu0).reshape(n_state, 1), np.array(gxu1).reshape(n_state, 1), np.array(gxu2).reshape(n_state, 1), np.array(gxu3).reshape(n_state, 1)))
            
            # A_mat = np.vstack((A_mat, np.array([0]*m_control*(n_state+m_control)).reshape(m_control, n_state + m_control))) * dt + np.eye(n_state + m_control)
                        
            # L_EKF, P = dynamics.EKF_gain(A_mat, C_EKF, P)
            
            # np.set_printoptions(2, linewidth=200)

            # # print(np.diag(P))
            
            # # print(L_EKF)

            # # print(asasas)
            # state_aug = torch.vstack((state[0, :].reshape(n_state, 1), gamma_actual_bs[0, :].reshape(m_control, 1)))
            # C_EKFtorch = torch.tensor(C_EKF,dtype=torch.float32)
            # state_EKF = torch.matmul(torch.tensor(A_mat, dtype=torch.float32), state_EKF) - torch.matmul(L_EKF, torch.matmul(C_EKFtorch, (state_EKF - state_aug)))
            
            # state_EKF = state_EKF + torch.vstack((fx_gamma.reshape(n_state, 1) * dt + torch.matmul(gx_gamma, u_nominal[0, :] * gamma_actual_bs[0, :]).reshape(n_state, 1) * dt, torch.zeros(m_control, 1))) - torch.matmul(L_EKF, torch.matmul(C_EKFtorch, (state_EKF - state_aug)))

            # h, grad_h = cbf.V_with_jacobian(state.reshape(n_sample, n_state, 1))
            
            u = u_nominal.clone()

            # u_gamma = u_nominal_gamma.clone()
            # u = util.fault_controller(u_nominal, fx, gx, h, grad_h)

            state_traj[:, k, :] = state.clone()
            
            state_traj_diff[:, k, :] = state_no_fault.clone() - state.clone()
            
            u_traj[:, k, :] = u.clone()

            # state_traj_gamma[:, k, :] = state_gamma.clone()
            gxu_no_fault = torch.matmul(gx, u.reshape(n_sample, m_control, 1))
            
            u = u * gamma_actual_bs
            
            gxu = torch.matmul(gx, u.reshape(n_sample, m_control, 1))

            dx = fx.reshape(n_sample, n_state) + gxu.reshape(n_sample, n_state)

            dx_no_fault = fx.reshape(n_sample, n_state) + gxu_no_fault.reshape(n_sample, n_state)
            
            state_no_fault = state.clone() + dx_no_fault * dt 

            state = state.clone() + dx * dt + torch.randn(n_sample, n_state) * dt
            
            # for j1 in range(n_sample):
            #     for j2 in range(n_state):
            #         if state[j1, j2] > sm[j2]:
            #             state[j1, j2] = sm[j2].clone()
            #         if state[j1, j2] < sl[j2]:
            #             state[j1, j2] = sl[j2].clone()

            is_safe = int(torch.sum(util.is_safe(state))) / n_sample

            safety_rate = (i * safety_rate + is_safe) / (i + 1)
        
        # print(state_EKF[-m_control:])
        
        dataset.add_data(state_traj.reshape(n_sample * traj_len, n_state), state_traj_diff.reshape(n_sample * traj_len, n_state), u_traj.reshape(n_sample * traj_len, m_control), torch.tensor([]).reshape(0, m_control))
        # print(t.toc())
        # gamma.to(torch.device('cpu'))
        # if gpu_id >= 0:
        #     gamma.to(torch.device(gpu_id))
        #     state_traj = state_traj.cuda(gpu_id)
        #     state_traj_diff = state_traj_diff.cuda(gpu_id)
        #     u_traj = u_traj.cuda(gpu_id)
        #     check_ind = np.mod(int(i * n_sample / 4), n_sample)
        #     gamma_fault = gamma(state_traj[check_ind, :, :].reshape(1, traj_len, n_state), state_traj_diff[check_ind, :, :].reshape(1, traj_len, n_state), u_traj[check_ind, :, :].reshape(1, traj_len, m_control))
        # print(gamma_fault)

        loss_np, acc_np = trainer.train_gamma()

        time_iter = t.tocvalue()
        print(
            'step, {}, loss, {:.3f}, acc, {:.3f}, safety rate, {:.3f}, time, {:.3f} '.format(
                i, loss_np, acc_np, safety_rate, time_iter))

        torch.save(gamma.state_dict(), str_data)
        
        if loss_np < 0.01 and loss_np < loss_current and i > 5:
            loss_current = loss_np.copy()
            torch.save(gamma.state_dict(), str_good_data)
        
        if loss_np < 0.001 and i > 250:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-fault_index', type=int, default=0)
    args = parser.parse_args()
    main(args)
