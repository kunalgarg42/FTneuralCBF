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
from trainer.NNfuncgrad_CF import CBF, alpha_param, NNController_new

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

init_param = 0  # int(input("use previous weights? (0 -> no, 1 -> yes): "))
print(init_param)

train_u = 0  # int(input("Train only CBF (0) or both CBF and u (1): "))
print(train_u)

n_sample = 10000

fault = nominal_params["fault"]

fault_control_index = 1

t = TicToc()


def main(args):
    fault = args.fault
    dynamics = CrazyFlies(x=x0, goal=xg, nominal_params=nominal_params, dt=dt)
    util = Utils(n_state=n_state, m_control=m_control, dyn=dynamics, params=nominal_params, fault=fault,
                 fault_control_index=fault_control_index)
    # nn_controller = NNController_new(n_state=n_state, m_control=m_control)
    cbf = CBF(dynamics=dynamics, n_state=n_state, m_control=m_control, fault=fault,
              fault_control_index=fault_control_index)
    # alpha = alpha_param(n_state=n_state)

    # nn_controller.load_state_dict(torch.load('./good_data/data/CF_controller_NN_weights.pth'))
    # nn_controller.eval()
    if init_param == 1:
        try:
            if fault == 0:
                cbf.load_state_dict(torch.load('./good_data/data/CF_cbf_NN_weightsCBF.pth'))
                # nn_controller.load_state_dict(torch.load('./good_data/data/CF_controller_NN_weights.pth'))
                # alpha.load_state_dict(torch.load('./good_data/data/CF_alpha_NN_weights.pth'))
            else:
                cbf.load_state_dict(torch.load('./good_data/data/CF_cbf_FT_weightsCBF.pth'))
                # nn_controller.load_state_dict(torch.load('./good_data/data/CF_controller_FT_weights.pth'))
                # alpha.load_state_dict(torch.load('./good_data/data/CF_alpha_FT_weights.pth'))
            cbf.eval()
            # nn_controller.eval()
            # alpha.eval()
        except:
            print("No good data available")
            try:
                if fault == 0:
                    cbf.load_state_dict(torch.load('./data/CF_cbf_NN_weightsCBF.pth'))
                    # nn_controller.load_state_dict(torch.load('./data/CF_controller_NN_weights.pth'))
                    # alpha.load_state_dict(torch.load('./data/CF_alpha_NN_weights.pth'))
                else:
                    cbf.load_state_dict(torch.load('./data/CF_cbf_FT_weightsCBF.pth'))
                    # nn_controller.load_state_dict(torch.load('./data/CF_controller_FT_weights.pth'))
                    # alpha.load_state_dict(torch.load('./data/CF_alpha_FT_weights.pth'))
                cbf.eval()
                # nn_controller.eval()
                # alpha.eval()
            except:
                print("No pre-train data available")

    dataset = Dataset_with_Grad(n_state=n_state, m_control=m_control, train_u=train_u)
    trainer = Trainer(cbf, dataset, n_state=n_state, m_control=m_control, j_const=2, dyn=dynamics,
                      dt=dt, action_loss_weight=0.001, params=nominal_params,
                      fault=fault, gpu_id=-1,
                      fault_control_index=fault_control_index)
    loss_np = 1.0
    safety_rate = 0.0
    goal_reached = 0.0

    sm, sl = dynamics.state_limits()
    safe_m, safe_l = dynamics.safe_limits(sm, sl)
    i_train = 0
    if train_u == 1:
        for i in range(int(config.TRAIN_STEPS / config.POLICY_UPDATE_INTERVAL)):
            # t.tic()
            # print(i)
            if init_add == 1:
                init_states0 = util.x_bndr(safe_m, safe_l, n_sample)
                init_states0 = init_states0.reshape(n_sample, n_state) + 1 * torch.normal(mean=(sm + sl) / 2,
                                                                                          std=torch.ones(n_state))
            else:
                init_states0 = torch.tensor([]).reshape(0, n_state)

            init_states1 = util.x_samples(sm, sl, n_sample)
            # init_states1 = init_states1.reshape(n_sample, n_state) + torch.randn(n_sample, n_state)
            # torch.normal(mean=(sm + sl) / 2,
            #
            #                                                                       std=torch.ones(n_state))
            init_states = torch.vstack((init_states0, init_states1))

            num_states = init_states.shape[0]

            for j in range(num_states):
                for k in range(n_state):
                    if init_states[j, k] < sl[k] * 0.5:
                        init_states[j, k] = sl[k].clone()
                    if init_states[j, k] > sm[k] * 2:
                        init_states[j, k] = sm[k].clone()

            dataset.add_data(init_states, torch.tensor([]).reshape(0, m_control),
                             torch.tensor([]).reshape(0, m_control))

            is_safe = int(torch.sum(util.is_safe(init_states))) / num_states

            safety_rate = (safety_rate * i + is_safe) / (i + 1)

            loss_np, acc_np, loss_h_safe, loss_h_dang, loss_alpha, loss_deriv_safe, loss_deriv_dang, loss_deriv_mid, loss_action = trainer.train_cbf_and_controller(
                eps=0.01, train_CF=1)
            print(
                'step, {}, loss, {:.3f}, safety rate, {:.3f}, goal reached, {:.3f}, acc, {}, '
                'loss_h_safe, {:.3f}, loss_h_dang, {:.3f}, loss_alpha, {:.3f}, loss_deriv_safe, {:.3f}, '
                'loss_deriv_dang, {:.3f}, loss_deriv_mid, {:.3f}, loss_action, {:.3f}'.format(
                    i, loss_np, safety_rate, goal_reached, acc_np, loss_h_safe, loss_h_dang, loss_alpha,
                    loss_deriv_safe, loss_deriv_dang, loss_deriv_mid, loss_action))
            if fault == 0:
                torch.save(cbf.state_dict(), './data/CF_cbf_NN_weights.pth')
                torch.save(nn_controller.state_dict(), './data/CF_controller_NN_weights.pth')
                torch.save(alpha.state_dict(), './data/CF_alpha_NN_weights.pth')
            else:
                torch.save(cbf.state_dict(), './data/CF_cbf_FT_weights.pth')
                torch.save(nn_controller.state_dict(), './data/CF_controller_FT_weights.pth')
                torch.save(alpha.state_dict(), './data/CF_alpha_FT_weights.pth')
            if loss_np < 0.01:
                if fault == 0:
                    torch.save(cbf.state_dict(), './good_data/data/CF_cbf_NN_weights.pth')
                    torch.save(nn_controller.state_dict(), './good_data/data/CF_controller_NN_weights.pth')
                    torch.save(alpha.state_dict(), './good_data/data/CF_alpha_NN_weights.pth')
                else:
                    torch.save(cbf.state_dict(), './good_data/data/CF_cbf_FT_weights.pth')
                    torch.save(nn_controller.state_dict(), './good_data/data/CF_controller_FT_weights.pth')
                    torch.save(alpha.state_dict(), './good_data/data/CF_alpha_FT_weights.pth')
            if loss_np < 0.001 and i > 100:
                break
    else:
        for i in range(int(config.TRAIN_STEPS / config.POLICY_UPDATE_INTERVAL)):
            t.tic()
            if init_add == 1:
                init_states0 = util.x_bndr(safe_m, safe_l, 2 * n_sample)
            else:
                init_states0 = torch.tensor([]).reshape(0, n_state)

            # init_states1 = util.x_samples(sm, sl, config.POLICY_UPDATE_INTERVAL)
            #
            # init_states = torch.vstack((init_states0, init_states1))
            init_states = init_states0.clone()

            # for j in range(5):
            #     # print(j)
            #     lower_bound = sl.clone() + (sm.clone() - sl.clone()) * j / 5
            #     upper_bound = sl.clone() + (sm.clone() - sl.clone()) * (j + 1) / 5

            #     init_states1 = util.x_samples(upper_bound, lower_bound, int(config.POLICY_UPDATE_INTERVAL / 5))

            #     init_states = torch.vstack((init_states, init_states1))
            
            safe_states = dynamics.sample_safe(n_sample)
            safe_states = safe_states.reshape(n_sample, n_state)

            unsafe_states = dynamics.sample_unsafe(n_sample)

            unsafe_states = unsafe_states.reshape(n_sample, n_state)

            init_states = torch.vstack((safe_states, unsafe_states))
            # unsafe_states = util.x_samples(sm, safe_m, n_sample)
            # init_states = torch.vstack((init_states, unsafe_states))

            num_states = init_states.shape[0]

            init_states = init_states + 0.1 * i * torch.randn(num_states, n_state) / 100

            dataset.add_data(init_states, torch.tensor([]).reshape(0, m_control),
                             torch.tensor([]).reshape(0, m_control))

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
                torch.save(cbf.state_dict(), './data/CF_cbf_NN_weightsCBF_FxTS.pth')
                # torch.save(alpha.state_dict(), './data/CF_alpha_NN_weightsCBF.pth')
            else:
                torch.save(cbf.state_dict(), './data/CF_cbf_FT_weightsCBF_FxTS.pth')
                # torch.save(alpha.state_dict(), './data/CF_alpha_FT_weightsCBF.pth')
            if loss_np < 0.01:
                if fault == 0:
                    torch.save(cbf.state_dict(), './good_data/data/CF_cbf_NN_weightsCBF_FxTS.pth')
                    # torch.save(alpha.state_dict(), './good_data/data/CF_alpha_NN_weightsCBF.pth')
                else:
                    torch.save(cbf.state_dict(), './good_data/data/CF_cbf_FT_weightsCBF_FxTS.pth')
                    # torch.save(alpha.state_dict(), './good_data/data/CF_alpha_FT_weightsCBF.pth')
            if loss_np < 0.001 and i > 100:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-fault', type=int, default=0)
    args = parser.parse_args()
    main(args)
