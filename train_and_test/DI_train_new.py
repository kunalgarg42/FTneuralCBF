import os
import sys
import torch
import numpy as np
import argparse

sys.path.insert(1, os.path.abspath('..'))
sys.path.insert(1, os.path.abspath('.'))

from pytictoc import TicToc
from dynamics.DI_dyn import DI
from trainer import config
from trainer.constraints_crazy import constraints
from trainer.datagen import Dataset_with_Grad
from trainer.trainer import Trainer
from trainer.utils import Utils
from trainer.NNfuncgrad_CF import CBF, alpha_param, NNController_new


dt = 0.01

m_control = 3

n_state = m_control * 2

x0 = torch.randn(1, n_state)

xg = torch.randn(1, n_state)

nominal_params = config.CRAZYFLIE_PARAMS

fault = nominal_params["fault"]
print(fault)

init_add = 0  # int(input("init data add? (0 -> no, 1 -> yes): "))
print(init_add)

init_param = 1  # int(input("use previous weights? (0 -> no, 1 -> yes): "))
print(init_param)

train_u = 0  # int(input("Train only CBF (0) or both CBF and u (1): "))
print(train_u)

n_sample = 10000

fault = nominal_params["fault"]

fault_control_index = 1

t = TicToc()

def main(args):
    fault = args.fault
    nominal_params["fault"] = fault
    dynamics = DI(x=x0, goal=xg, dim=m_control, nominal_parameters=nominal_params, dt=dt)
    util = Utils(n_state=n_state, m_control=m_control, dyn=dynamics, params=nominal_params, fault=fault,
                 fault_control_index=fault_control_index)
    cbf = CBF(dynamics=dynamics, n_state=n_state, m_control=m_control, fault=fault,
              fault_control_index=fault_control_index)
    if init_param == 1:
        try:
            if fault == 0:
                cbf.load_state_dict(torch.load('./good_data/data/DI_cbf_NN_weightsCBF.pth'))
            else:
                cbf.load_state_dict(torch.load('./good_data/data/DI_cbf_FT_weightsCBF.pth'))
            cbf.eval()
        except:
            print("No good data available")
            try:
                if fault == 0:
                    cbf.load_state_dict(torch.load('./data/DI_cbf_NN_weightsCBF.pth'))
                else:
                    cbf.load_state_dict(torch.load('./data/DI_cbf_FT_weightsCBF.pth'))
                cbf.eval()
            except:
                print("No pre-train data available")

    dataset = Dataset_with_Grad(n_state=n_state, m_control=m_control, train_u=train_u)
    trainer = Trainer(cbf, dataset, n_state=n_state, m_control=m_control, j_const=2, dyn=dynamics,
                      dt=dt, action_loss_weight=0.001, params=nominal_params, traj_len=100,
                      fault=fault, gpu_id=0,
                      fault_control_index=fault_control_index)
    loss_np = 1.0
    safety_rate = 0.0
    goal_reached = 0.0

    sm, sl = dynamics.state_limits()
    safe_m, safe_l = dynamics.safe_limits()
    i_train = 0
    loss_current = 100.0
    for i in range(int(config.TRAIN_STEPS / config.POLICY_UPDATE_INTERVAL)):
        t.tic()
        if init_add == 1:
            init_states0 = util.x_bndr(safe_m, safe_l, 2 * n_sample) + torch.randn(2 * n_sample, n_state) * 5
        else:
            init_states0 = torch.tensor([]).reshape(0, n_state)

        init_states = init_states0.clone()
        
        safe_states = dynamics.sample_safe(n_sample)

        safe_states = safe_states.reshape(n_sample, n_state)

        unsafe_states = dynamics.sample_unsafe(n_sample)

        unsafe_states = unsafe_states.reshape(n_sample, n_state)

        # mid_states = dynamics.sample_mid(n_sample)

        # mid_states = mid_states.reshape(n_sample, n_state)

        init_states = torch.vstack((init_states, safe_states, unsafe_states))

        num_states = init_states.shape[0]

        init_states = init_states + torch.randn(num_states, n_state) / 100 * i
                
        dataset.add_data(init_states, torch.tensor([]).reshape(0, n_state),
                            torch.tensor([]).reshape(0, m_control),torch.tensor([]).reshape(0, m_control))

        is_safe = int(torch.sum(util.is_safe(init_states))) / num_states

        safety_rate = (i * safety_rate + is_safe) / (i + 1)
        
        if loss_np < 0.01 or i_train >= i - 1:
            i_train = int(config.TRAIN_STEPS / config.POLICY_UPDATE_INTERVAL) / 2 + 1
        else:
            i_train = i

        loss_np, acc_np, loss_h_safe, loss_h_dang, loss_deriv_safe, loss_deriv_dang, loss_deriv_mid = trainer.train_cbf()
        time_iter = t.tocvalue()
        print(
            'step, {}, loss, {:.3f}, safety rate, {:.3f}, goal reached, {:.3f}, acc, {}, '
            'loss_h_safe, {:.3f}, loss_h_dang, {:.3f}, loss_deriv_safe, {:.3f}, '
            'loss_deriv_dang, {:.3f}, loss_deriv_mid, {:.3f}, time, {:.3f} '.format(
                i, loss_np, safety_rate, goal_reached, acc_np, loss_h_safe, loss_h_dang,
                loss_deriv_safe, loss_deriv_dang, loss_deriv_mid, time_iter))
        if fault == 0:
            torch.save(cbf.state_dict(), './data/DI_cbf_NN_weightsCBF.pth')
        else:
            torch.save(cbf.state_dict(), './data/DI_cbf_FT_weightsCBF.pth')
        if loss_np < 0.01 and loss_np < loss_current and i > 50:
            loss_current = loss_np.copy()
            if fault == 0:
                torch.save(cbf.state_dict(), './good_data/data/DI_cbf_NN_weightsCBF.pth')
            else:
                torch.save(cbf.state_dict(), './good_data/data/DI_cbf_FT_weightsCBF.pth')
        if loss_np < 0.001 and i > 100:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-fault', type=int, default=0)
    args = parser.parse_args()
    main(args)
