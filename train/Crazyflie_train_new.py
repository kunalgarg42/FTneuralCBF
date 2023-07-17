import os
import sys
import torch
import numpy as np
import argparse
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
from trainer.NNfuncgrad_CF import CBF, NNController_new

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

n_sample = 10000

fault = nominal_params["fault"]

fault_control_index = 1

t = TicToc()

gpu_id = 0 # torch.cuda.current_device()

if platform.uname()[1] == 'realm2':
    gpu_id = 3

def main(args):
    fault = args.fault
    nominal_params["fault"] = fault
    dt = args.dt

    if platform.uname()[1] == 'realm2':
        gpu_id = args.gpu

        if gpu_id >= 0:
            use_cuda = True
        else:
            use_cuda = False

        if gpu_id >= 0:
            device = torch.device(args.gpu if use_cuda else 'cpu')
        else:
            device = torch.device('cpu')
        print(f'> Training with {device}')

    else:
        gpu_id = args.gpu
        use_cuda = torch.cuda.is_available() and not args.cpu
        if use_cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = torch.device('cuda' if use_cuda else 'cpu')
        print(f'> Training with {device}')

    dynamics = CrazyFlies(x=x0, goal=xg, nominal_params=nominal_params, dt=dt)
    util = Utils(n_state=n_state, m_control=m_control, dyn=dynamics, params=nominal_params, fault=fault,
                 fault_control_index=fault_control_index)
    nn_controller = NNController_new(n_state=n_state, m_control=m_control)
    cbf = CBF(dynamics=dynamics, n_state=n_state, m_control=m_control, fault=fault,
              fault_control_index=fault_control_index)

    if init_param == 1:
        try:
            if fault == 0:
                cbf.load_state_dict(torch.load('./good_data/data/CF_cbf_NN_weightsCBF_with_u_new.pth'))
                nn_controller.load_state_dict(torch.load('./good_data/data/CF_controller_NN_weights_new.pth'))
            else:
                cbf.load_state_dict(torch.load('./good_data/data/CF_cbf_FT_weightsCBF_with_u_new.pth'))
                nn_controller.load_state_dict(torch.load('./good_data/data/CF_controller_FT_weights_new.pth'))
                # nn_controller.load_state_dict(torch.load('./good_data/data/CF_controller_FT_weights.pth'))
                # alpha.load_state_dict(torch.load('./good_data/data/CF_alpha_FT_weights.pth'))
            cbf.eval()
            nn_controller.eval()
            # alpha.eval()
        except:
            print("No good data available")
            try:
                if fault == 0:
                    cbf.load_state_dict(torch.load('./data/CF_cbf_NN_weightsCBF_with_u_new.pth'))
                    nn_controller.load_state_dict(torch.load('./data/CF_controller_NN_weights_new.pth'))
                    # nn_controller.load_state_dict(torch.load('./data/CF_controller_NN_weights.pth'))
                    # alpha.load_state_dict(torch.load('./data/CF_alpha_NN_weights.pth'))
                else:
                    cbf.load_state_dict(torch.load('./data/CF_cbf_FT_weightsCBF_with_u_new.pth'))
                    nn_controller.load_state_dict(torch.load('./data/CF_controller_FT_weights_new.pth'))
                    # nn_controller.load_state_dict(torch.load('./data/CF_controller_FT_weights.pth'))
                    # alpha.load_state_dict(torch.load('./data/CF_alpha_FT_weights.pth'))
                cbf.eval()
                nn_controller.eval()
                # alpha.eval()
            except:
                print("No pre-train data available")

    dataset = Dataset_with_Grad(y_state=n_state, n_state=n_state, m_control=m_control, train_u=1, buffer_size=n_sample*20000, traj_len=1)
    trainer = Trainer(cbf, nn_controller, dataset, gamma=None, n_state=n_state, m_control=m_control, j_const=2, dyn=dynamics,
                      dt=dt, action_loss_weight=0.001, params=nominal_params,
                      fault=fault, gpu_id=gpu_id, num_traj=n_sample, traj_len=0,
                      fault_control_index=fault_control_index, model_factor=0, device=device)
    loss_np = 1.0
    safety_rate = 0.0
    goal_reached = 0.0

    sm, sl = dynamics.state_limits()
    safe_m, safe_l = dynamics.safe_limits(sm, sl, fault)
    
    loss_current = 100.0
    for i in range(int(config.TRAIN_STEPS / config.POLICY_UPDATE_INTERVAL)):
        new_goal = dynamics.sample_safe(1)

        new_goal = new_goal.reshape(n_state, 1)

        t.tic()
        if init_add == 1:
            init_states0 = util.x_bndr(safe_m, safe_l, 2 * n_sample) + torch.randn(2 * n_sample, n_state) * 2
        else:
            init_states0 = torch.tensor([]).reshape(0, n_state)

        init_states = init_states0.clone()
        
        safe_states = dynamics.sample_safe(n_sample)

        safe_states = safe_states.reshape(n_sample, n_state)

        unsafe_states = dynamics.sample_unsafe(n_sample)

        unsafe_states = unsafe_states.reshape(n_sample, n_state)

        init_states = torch.vstack((init_states, safe_states, unsafe_states))

        num_states = init_states.shape[0]

        init_states = init_states + torch.randn(num_states, n_state) / 100 * i

        dataset.add_data(init_states, torch.tensor([]).reshape(0, n_state), torch.tensor([]).reshape(0, m_control),
                            torch.tensor([]).reshape(0, m_control))

        is_safe = int(torch.sum(util.is_safe(init_states))) / num_states

        safety_rate = (i * safety_rate + is_safe) / (i + 1)

        loss_np, acc_np, loss_h_safe, loss_h_dang, loss_deriv_safe, loss_deriv_dang, loss_deriv_mid = trainer.train_cbf_and_u(goal=new_goal)
        time_iter = t.tocvalue()
        print(
            'step, {}, loss, {:.3f}, safety rate, {:.3f}, goal reached, {:.3f}, acc, {}, '
            'loss_h_safe, {:.3f}, loss_h_dang, {:.3f}, loss_deriv_safe, {:.3f}, '
            'loss_deriv_dang, {:.3f}, loss_deriv_mid, {:.3f}, time, {:.3f} '.format(
                i, loss_np, safety_rate, goal_reached, acc_np, loss_h_safe, loss_h_dang,
                loss_deriv_safe, loss_deriv_dang, loss_deriv_mid, time_iter))
        if loss_np <= loss_current and i > 5:
            loss_current = loss_np.copy()
            if fault == 0:
                torch.save(cbf.state_dict(), './data/CF_cbf_NN_weightsCBF_with_u_new.pth')
                torch.save(nn_controller.state_dict(), './data/CF_controller_NN_weights_new.pth')
            else:
                torch.save(cbf.state_dict(), './data/CF_cbf_FT_weightsCBF_with_u_new.pth')
                torch.save(nn_controller.state_dict(), './data/CF_controller_FT_weights_new.pth')
        if loss_np < 0.01 and loss_np < loss_current and i > 50:
            loss_current = loss_np.copy()
            if fault == 0:
                torch.save(cbf.state_dict(), './good_data/data/CF_cbf_NN_weightsCBF_with_u_new.pth')
                torch.save(nn_controller.state_dict(), './good_data/data/CF_controller_NN_weights_new.pth')
            else:
                torch.save(cbf.state_dict(), './good_data/data/CF_cbf_FT_weightsCBF_with_u_new.pth')
                torch.save(nn_controller.state_dict(), './good_data/data/CF_controller_FT_weights_new.pth')
        if loss_np < 0.001 and i > 500:
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fault', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dt', type=float, default=0.001)
    parser.add_argument('--cpu', type=bool, default=False)
    args = parser.parse_args()
    main(args)
