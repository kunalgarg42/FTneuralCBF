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
from trainer.trainer_FxTS import Trainer
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

rg = 1.2

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
                      fault=fault, gpu_id=0,
                      goal=xg, rg=rg,
                      fault_control_index=fault_control_index)
    loss_np = 1.0
    safety_rate = 0.0
    goal_reached = 0.0

    sm, sl = dynamics.state_limits()
    safe_m, safe_l = dynamics.safe_limits(sm, sl)

    i_train = 0

    for i in range(int(config.TRAIN_STEPS / config.POLICY_UPDATE_INTERVAL)):
        t.tic()

        init_states = util.x_samples(sm, sl, config.POLICY_UPDATE_INTERVAL)

        num_states = init_states.shape[0]

        num_half = int(num_states / 2)

        init_states = init_states + i * torch.randn(num_states, n_state) / 100

        init_states[:, 2] = xg[0, 2] * torch.ones(num_states,) + torch.randn(num_states,)

        init_states[-num_half:, 2] = xg[0, 2] * torch.ones(num_half,) + torch.randn(num_half,) * 5

        safe_mask, _ = trainer.get_mask_cpu(init_states, xg, rg)

        print(torch.sum(safe_mask))

        dataset.add_data(init_states, torch.tensor([]).reshape(0, m_control),
                         torch.tensor([]).reshape(0, m_control))

        loss_np, acc_np, loss_V_in, loss_V_out, loss_dotV_in, loss_dotV_out = trainer.train_cbf()
        time_iter = t.tocvalue()
        print(
            'step, {}, loss, {:.3f}, acc, {}, loss_V_in, {:.3f}, loss_V_out, {:.3f}, loss_dot_V_in, {:.3f}, '
            'loss_dot_V_out, {:.3f}, time, {:.3f} '.format(
                i, loss_np, acc_np, loss_V_in, loss_V_out, loss_dotV_in, loss_dotV_out, time_iter))
        # if fault == 0:
        torch.save(cbf.state_dict(), './data/CF_FxTSCLF_weights.pth')
        # torch.save(alpha.state_dict(), './data/CF_alpha_NN_weightsCBF.pth')
        # else:
        #     torch.save(cbf.state_dict(), './data/CF_FxTSCLF_FT_weightsCF_FxTS.pth')
        #     # torch.save(alpha.state_dict(), './data/CF_alpha_FT_weightsCBF.pth')
        if loss_np < 0.01:
            # if fault == 0:
            torch.save(cbf.state_dict(), './good_data/data/CF_FxTSCLF_weights.pth')
            # torch.save(alpha.state_dict(), './good_data/data/CF_alpha_NN_weightsCBF.pth')
            # else:
            #     torch.save(cbf.state_dict(), './good_data/data/CF_cbf_FT_weightsCBF_FxTS.pth')
            # torch.save(alpha.state_dict(), './good_data/data/CF_alpha_FT_weightsCBF.pth')
        if loss_np < 0.001 and i > 100:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-fault', type=int, default=0)
    args = parser.parse_args()
    main(args)
