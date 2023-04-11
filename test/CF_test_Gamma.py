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

init_param = 1

n_sample = 500

traj_len = 100

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

    sm, sl = dynamics.state_limits()
    safe_m, safe_l = dynamics.safe_limits(sm, sl, fault)
    
    gamma_actual_bs = torch.ones(n_sample, m_control)
    fault_value = 0.0
    for j in range(n_sample):
        fault_control_index = np.mod(j, 5)
        if fault_control_index < 4:
            gamma_actual_bs[j, fault_control_index] = fault_value
        

    state0 = util.x_samples(safe_m, safe_l, n_sample)
    
    state_traj = torch.zeros(n_sample, traj_len, n_state)
    
    state_traj_diff = state_traj.clone()

    u_traj = torch.zeros(n_sample, traj_len, m_control)
    
    state = state0.clone()
    
    state_no_fault = state0.clone()

    u_nominal = dynamics.u_nominal(state)
    
    t.tic()

    for k in range(traj_len):
        
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

    gamma_NN = gamma(state_traj.reshape(n_sample, traj_len, n_state), (1 - args.fault_index) * state_traj_diff.reshape(n_sample, traj_len, n_state), u_traj.reshape(n_sample, traj_len, m_control))
    
    # print(gamma_NN[-4:, :])

    gamma_pred = gamma_NN.clone().detach()
    for j in range(n_sample):
        min_gamma = int(torch.argmin(gamma_pred[j, :]))
        # print(min_gamma)
        for i in range(m_control):
            if i == min_gamma and gamma_pred[j, min_gamma] < fault_value + 0.2:
                continue
            else:
                gamma_pred[j, i] = 1.0
    # print(gamma_pred[-4:, :])
    
    # print(gamma_actual_bs)
    gamma_err = gamma_pred - gamma_actual_bs
    correct_gamma = 1 - torch.sum(torch.linalg.norm(gamma_err, dim=1)) / n_sample
    print(correct_gamma)
    # time_iter = t.tocvalue()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-fault_index', type=int, default=0)
    args = parser.parse_args()
    main(args)
