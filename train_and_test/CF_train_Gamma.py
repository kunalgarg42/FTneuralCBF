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

dt = 0.01
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

n_sample = 1000

traj_len = 50

fault = nominal_params["fault"]

fault_control_index = 0

str_data = './data/CF_gamma_NN_weights{}.pth'.format(fault_control_index)
str_good_data = './good_data/data/CF_gamma_NN_weights{}.pth'.format(fault_control_index)

t = TicToc()


def main(args):
    fault = 1
    fault_control_index = args.fault_index
    nominal_params["fault"] = fault
    dynamics = CrazyFlies(x=x0, goal=xg, nominal_params=nominal_params, dt=dt)
    util = Utils(n_state=n_state, m_control=m_control, dyn=dynamics, params=nominal_params, fault=fault,
                 fault_control_index=fault_control_index)
    cbf = CBF(dynamics=dynamics, n_state=n_state, m_control=m_control, fault=fault,
              fault_control_index=fault_control_index)
    gamma = Gamma(n_state=n_state, m_control=m_control)

    if init_param == 1:
        try:
            gamma.load_state_dict(torch.load(str_good_data))
            cbf.eval()
        except:
            print("No good data available")
            try:
                gamma.load_state_dict(torch.load(str_data))
                gamma.eval()
            except:
                print("No pre-train data available")
    
    cbf.load_state_dict(torch.load('./good_data/data/CF_cbf_NN_weightsCBF.pth'))
    cbf.eval()

    dataset = Dataset_with_Grad(n_state=n_state, m_control=m_control, train_u=train_u, buffer_size=n_sample*traj_len)
    trainer = Trainer(cbf, dataset, gamma=gamma, n_state=n_state, m_control=m_control, j_const=2, dyn=dynamics,
                      dt=dt, action_loss_weight=0.001, params=nominal_params,
                      fault=fault, gpu_id=0, num_traj=n_sample,
                      fault_control_index=fault_control_index)
    loss_np = 1.0
    safety_rate = 0.0

    sm, sl = dynamics.state_limits()
    safe_m, safe_l = dynamics.safe_limits(sm, sl, fault)
    i_train = 0
    
    loss_current = 100.0
    gamma_actual = torch.ones(1, 4)
    
    gamma_actual[fault_control_index, 0] = 0.0

    gamma_actual_bs = gamma_actual.repeat(n_sample, 1)

    for i in range(100):
        state0 = util.x_samples(safe_m, safe_l, n_sample)
        
        state = state0.clone()

        u_nominal = dynamics.u_nominal(state)
        gamma = gamma.to(torch.device('cpu'))
        gamma_fault = gamma(state, u_nominal)
        
        t.tic()
        
        for k in range(traj_len):
            
            u_nominal = dynamics.u_nominal(state)

            fx = dynamics._f(state, params=nominal_params)
            gx = dynamics._g(state, params=nominal_params)

            h, grad_h = cbf.V_with_jacobian(state.reshape(n_sample, n_state, 1))
            
            u = util.fault_controller(u_nominal, fx, gx, h, grad_h)

            dataset.add_data(state, torch.tensor(u), torch.tensor([]).reshape(0, m_control))

            u = u * gamma_actual_bs

            gxu = torch.matmul(gx, u.reshape(n_sample, m_control, 1))

            dx = fx.reshape(n_sample, n_state) + gxu.reshape(n_sample, n_state)

            state = state.clone() + dx * dt

            gamma_fault = gamma_fault + gamma(state, u) * dt

            for j1 in range(n_sample):
                for j2 in range(n_state):
                    if state[j1, j2] > sm[j2]:
                        state[j1, j2] = sm[j2].clone()
                    if state[j1, j2] < sl[j2]:
                        state[j1, j2] = sl[j2].clone()

            is_safe = int(torch.sum(util.is_safe(state))) / n_sample

            safety_rate = (i * safety_rate + is_safe) / (i + 1)
            
            if loss_np < 0.01 or i_train >= i - 1:
                i_train = int(config.TRAIN_STEPS / config.POLICY_UPDATE_INTERVAL) / 2 + 1
            else:
                i_train = i
        
        print(gamma_fault[-1, :])

        loss_np = trainer.train_gamma(gamma_actual, traj_len)
        time_iter = t.tocvalue()
        print(
            'step, {}, loss, {:.3f}, safety rate, {:.3f}, time, {:.3f} '.format(
                i, loss_np, safety_rate, time_iter))

        torch.save(gamma.state_dict(), str_data)
        
        if loss_np < 0.01 and loss_np < loss_current and i > 50:
            loss_current = loss_np.copy()
            torch.save(gamma.state_dict(), str_good_data)
        
        if loss_np < 0.001 and i > 100:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-fault_index', type=int, default=0)
    args = parser.parse_args()
    main(args)
