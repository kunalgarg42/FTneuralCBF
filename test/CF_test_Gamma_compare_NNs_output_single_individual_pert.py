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
from trainer.NNfuncgrad_CF import CBF, Gamma_linear_deep_nonconv_output_single, Gamma_linear_LSTM_output_single

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

y_state = 6

m_control = 4

nominal_params = config.CRAZYFLIE_PARAMS

fault = nominal_params["fault"]

use_good = 1

traj_len = 100

Eval_steps = 199

fault = nominal_params["fault"]

t = TicToc()

gpu_id = 0

gamma_type = 'LSTM'

def main(args):
    fault_control_index = args.fault_index
    use_nom = args.use_nom
    model_factor = args.use_model
    dt = args.dt

    gpu_id = args.gpu
    use_cuda = torch.cuda.is_available() and not args.cpu
    if use_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    else:
        gpu_id = -1
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f'> Testing with {device}')

    acc_final = torch.zeros(4, 11, traj_len)

    use_previos_data = 1

    gamma_type = 'LSTM'

    if use_previos_data == 1:
        try:
            acc_final = torch.load('./log_files/acc_output_model_single_ind_pert.pt')
        except:
            use_previos_data = 0
    
    if use_previos_data == 0:
        print('Generating new data')
        for gamma_iter in tqdm.trange(4):
            if gamma_iter == 0 or gamma_iter == 2:
                nominal_params = config.CRAZYFLIE_PARAMS
            else:
                nominal_params = config.CRAZYFLIE_PARAMS_PERT_2
 
            if gamma_iter == 0 or gamma_iter == 1:
                model_factor == 0
            else:
                model_factor == 1

            dynamics = CrazyFlies(x=x0, goal=xg, nominal_params=nominal_params, dt=dt)
            util = Utils(n_state=n_state, m_control=m_control, dyn=dynamics, params=nominal_params, fault=fault,
                        fault_control_index=fault_control_index)
            cbf = CBF(dynamics=dynamics, n_state=n_state, m_control=m_control, fault=fault,
                    fault_control_index=fault_control_index)
            try:
                cbf.load_state_dict(torch.load('./supercloud_data/CF_cbf_NN_weightsCBF_with_u.pth'))
            except:
                try:
                    cbf.load_state_dict(torch.load('./good_data/data/CF_cbf_NN_weightsCBF_with_u.pth'))
                except:
                    cbf.load_state_dict(torch.load('./data/CF_cbf_NN_weightsCBF_with_u.pth'))
            cbf.eval()

            sm, sl = dynamics.state_limits()

            if use_nom == 1:
                n_sample = 22000
            else:
                n_sample = 1100
            
            if use_nom == 1:
                nsample_factor = 1
            else:
                nsample_factor = 1

            ind_y = torch.tensor([1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1]).bool()

            n_sample_iter = int(n_sample / nsample_factor)
    
            if gamma_type == 'LSTM':
                str_data = './data/CF_gamma_LSTM_output_single_' + str(y_state) + '_model_' + str(model_factor) + '_rates_sigmoid.pth'
                str_good_data = './good_data/data/CF_gamma_LSTM_output_single_' + str(y_state) + '_model_' + str(model_factor) + '_rates_sigmoid.pth'
                str_supercloud_data = './supercloud_data/CF_gamma_LSTM_output_single_' + str(y_state) + '_model_' + str(model_factor) + '_rates_sigmoid.pth'
            elif gamma_type == 'deep':
                str_data = './data/CF_gamma_deep_output_single_' + str(y_state) + '_model_' + str(model_factor) + '_rates_sigmoid.pth'
                str_good_data = './good_data/data/CF_gamma_deep_output_single_' + str(y_state) + '_model_' + str(model_factor) + '_rates_sigmoid.pth'
                str_supercloud_data = './supercloud_data/CF_gamma_deep_output_single_' + str(y_state) + '_model_' + str(model_factor) + '_rates_sigmoid.pth'
            else:
                NotImplementedError
            
            if gamma_type == 'deep':
                gamma = Gamma_linear_deep_nonconv_output_single(y_state=y_state, m_control=m_control, traj_len=traj_len, model_factor=model_factor)
            elif gamma_type == 'LSTM':
                gamma = Gamma_linear_LSTM_output_single(y_state=y_state, m_control=m_control, model_factor=model_factor)
            else:
                NotImplementedError
            
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
            
            gamma.eval().to(device)

            if gamma_iter % 2 == 0:
                gamma_actual_bs = torch.ones(n_sample_iter, m_control)

                for j in range(n_sample):
                    temp_var = np.mod(j, 11)
                    # if temp_var < 10:
                    gamma_actual_bs[j, fault_control_index] = temp_var / 10.0

                rand_ind = torch.randperm(n_sample_iter)

                gamma_actual_bs = gamma_actual_bs[rand_ind, :]
           
                state0 = dynamics.sample_safe(n_sample_iter // 11) + torch.randn(n_sample_iter // 11, n_state) * 2

                state0 = state0.repeat_interleave(11, dim=0)

                state0 = state0[rand_ind, :]

                new_goal = dynamics.sample_safe(1)

            state_traj = torch.zeros(n_sample_iter, Eval_steps, n_state) 

            state_traj_diff = state_traj.clone()   

            u_traj = torch.zeros(n_sample_iter, Eval_steps, m_control)
            
            state = state0.clone()

            state_no_fault = state.clone()

            for k in range(n_state):
                if k > 5:
                    state[:, k] = torch.clamp(state[:, k], sm[k] / 10, sl[k] / 10)
            
            u_nominal = dynamics.u_nominal(state)
            
            t.tic()

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
            
                state_no_fault = state.clone() + dx_no_fault * dt

                state = state.clone() + dx * dt
            
                for j2 in range(n_state):
                    ind_sm = state[:, j2] > sm[j2]
                    if torch.sum(ind_sm) > 0:
                        state[ind_sm, j2] = sm[j2].repeat(torch.sum(ind_sm),)
                    ind_sl = state[:, j2] < sl[j2]
                    if torch.sum(ind_sl) > 0:
                        state[ind_sl, j2] = sl[j2].repeat(torch.sum(ind_sl),)

                if k >= traj_len - 1:
                    if model_factor == 0:
                        gamma_NN = gamma(state_traj[:, k - traj_len + 1:k + 1, ind_y].to(device), u_traj[:, k - traj_len + 1:k + 1, :].to(device))
                    else:
                        state_data = torch.cat((state_traj[:, k - traj_len + 1:k + 1, ind_y], state_traj_diff[:, k-traj_len + 1:k+1, ind_y]), dim=-1)
                        gamma_NN = gamma(state_data.to(device), u_traj[:, k - traj_len + 1:k + 1, :].to(device))

                    gamma_pred = gamma_NN.reshape(n_sample_iter, m_control).clone().detach().cpu()

                    for j in range(11):
                        
                        index_fault = gamma_actual_bs[:, fault_control_index]  == j / 10

                        index_num = torch.sum(index_fault.float())

                        acc_final[gamma_iter, j, k-traj_len+1] =  torch.sum((torch.abs(gamma_pred[index_fault, fault_control_index] - gamma_actual_bs[index_fault, fault_control_index]) < 0.05).float()) / (index_num + 1e-5)
                        
        torch.save(acc_final, './log_files/acc_output_model_single_ind_pert.pt')
    else:
        print('Using previous data')
        
    # plt.rcParams['text.usetex'] = True
    fig = plt.figure(figsize=(20, 20))
    ax = fig.subplots(2, 1)
    
    plot_name = './plots/' + 'CF_gamma_compare_output_model_single_ind_pert.png'

    step = np.arange(0, traj_len, 1)

    markers_on = np.arange(0, step[-1], 10)

    # 6 shades of blue from light to dark
    colors = plt.cm.Blues(np.linspace(1, 0.5, 6))
    # 5 shades of red from light to dark
    colors = np.vstack((colors, plt.cm.Reds(np.linspace(0.5, 1, 5))))


    for gamma_iter1 in range(4):
        if gamma_iter1 == 0 or gamma_iter1 == 2:
            param = 'Nominal'
        else:
            param = 'Perturbed'
        if gamma_iter1 == 0 or gamma_iter1 == 1:
            model_factor = 0
        else:
            model_factor = 1
        gamma_iter = int(gamma_iter1 > 1)
        # if gamma_iter == 0:
        #     gamma_type = 'LSTM'
        # elif gamma_iter == 1:
        #     gamma_type = 'Linear MLP'
        for k in [0, 5, 10]:
            acc_fail = acc_final[gamma_iter1, k, :]
            if gamma_iter1 == 0 or gamma_iter1 == 2:
                ax[gamma_iter].plot(step, acc_fail, color = colors[k], linestyle='-', label = '$\Theta$: ' + str(round(k / 10, 2)) + ' [' + param + ']', marker="o", markevery=markers_on,markersize=10, linewidth=3)
            else:
                ax[gamma_iter].plot(step, acc_fail, color = colors[k], linestyle='--', label = '$\Theta$: ' + str(round(k / 10, 2)) + ' [' + param + ']', marker="^", markevery=markers_on,markersize=10, linewidth=3)
        if gamma_iter == 1:    
            ax[gamma_iter].set_xlabel('Length of trajectory with failed actuator', fontsize = 40)
        if model_factor == 0:
            ax[gamma_iter].set_ylabel('Accuracy: [Model Free]', fontsize = 40)
        else:
            ax[gamma_iter].set_ylabel('Accuracy: [Model Based]', fontsize = 40)
        
        
        # plt.title('Failure Test Accuracy', fontsize = 20)
        ax[gamma_iter].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 40)
        ax[gamma_iter].set_xlim(step[0], step[-1])
        if gamma_iter == 0:
            ax[gamma_iter].set_ylim(0.0, 1.1)
        else:
            ax[gamma_iter].set_ylim(0.0, 1.1)
        ax[gamma_iter].tick_params(axis = "x", labelsize = 30)
        ax[gamma_iter].tick_params(axis = "y", labelsize = 30)
    
    plt.tight_layout()

    plt.savefig(plot_name)

    print("saved file:", plot_name) 
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-fault_index', type=int, default=1)
    parser.add_argument('-traj_len', type=int, default=100)
    parser.add_argument('-use_model', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cpu', type=bool, default=False)
    parser.add_argument('--dt', type=float, default=0.002)
    parser.add_argument('--use_nom', type =int, default=1)
    args = parser.parse_args()
    main(args)
