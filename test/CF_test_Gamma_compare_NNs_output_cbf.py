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
from trainer.NNfuncgrad_CF import CBF, Gamma_linear_deep_nonconv_output, Gamma_linear_LSTM_output

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

gpu_id = 3

gamma_type = 'LSTM'

def main(args):
    use_saved_data = args.use_saved_data
    fault_control_index = args.fault_index
    use_nom = args.use_nom
    model_factor = 1
    dt = args.dt
    rates = args.rates

    gamma_type = args.gamma_type

    if use_saved_data == 1:
        try:
            if rates == 0:
                acc_final = torch.load('./log_files/acc_output_model_' + str(model_factor) + '_'+ gamma_type + '_cbf.pt')
            else:
                acc_final = torch.load('./log_files/acc_output_model_' + str(model_factor) + '_' + gamma_type + '_rates_cbf.pt')
        except:
            use_saved_data = 0
    else:
        acc_final = torch.zeros(4, 2, traj_len)

    if use_saved_data == 0:
        
        acc_final = torch.zeros(4, 2, traj_len)

        print('Generating new accuracy data')
        if rates == 0:
            ind_y = torch.tensor([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]).bool()
        else:
            ind_y = torch.tensor([1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1]).bool()

        gpu_id = args.gpu
        use_cuda = torch.cuda.is_available() and not args.cpu
        if use_cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        else:
            gpu_id = -1
        device = torch.device('cuda' if use_cuda else 'cpu')
        print(f'> Testing with {device}')


        dynamics = CrazyFlies(x=x0, goal=xg, nominal_params=nominal_params, dt=dt)
        util = Utils(n_state=n_state, m_control=m_control, dyn=dynamics, params=nominal_params, fault=fault,
                    fault_control_index=fault_control_index)
        cbf = CBF(dynamics=dynamics, n_state=n_state, m_control=m_control, fault=fault,
                fault_control_index=fault_control_index)
        try:
            if gpu_id >= 0:
                cbf.load_state_dict(torch.load('./supercloud_data/CF_cbf_NN_weightsCBF_with_u.pth'))
            else:
                cbf.load_state_dict(torch.load('./supercloud_data/CF_cbf_NN_weightsCBF_with_u.pth', map_location=torch.device('cpu')))
        except:
            try:
                if gpu_id >= 0:
                    cbf.load_state_dict(torch.load('./good_data/data/CF_cbf_NN_weightsCBF_with_u.pth'))
                else:
                    cbf.load_state_dict(torch.load('./good_data/data/CF_cbf_NN_weightsCBF_with_u.pth', map_location=torch.device('cpu')))
            except:
                if gpu_id >= 0:
                    cbf.load_state_dict(torch.load('./data/CF_cbf_NN_weightsCBF_with_u.pth'))
                else:
                    cbf.load_state_dict(torch.load('./data/CF_cbf_NN_weightsCBF_with_u.pth', map_location=torch.device('cpu')))
        
        cbf.eval()

        sm, sl = dynamics.state_limits()
        

        for gamma_iter in tqdm.trange(2):
            for i in range(2):
                model_factor = i
                if rates == 0:
                    if gamma_type == 'LSTM':
                        str_data = './data/CF_gamma_LSTM_output' + str(y_state) + '_model_' + str(model_factor) + '_sigmoid.pth'
                        str_good_data = './good_data/data/CF_gamma_LSTM_output' + str(y_state) + '_model_' + str(model_factor) + '_sigmoid.pth'
                        str_supercloud_data = './supercloud_data/CF_gamma_LSTM_output' + str(y_state) + '_model_' + str(model_factor) + '_sigmoid.pth'
                    elif gamma_type == 'deep':
                        str_data = './data/CF_gamma_deep_output' + str(y_state) + '_model_' + str(model_factor) + '_sigmoid.pth'
                        str_good_data = './good_data/data/CF_gamma_deep_output' + str(y_state) + '_model_' + str(model_factor) + '_sigmoid.pth'
                        str_supercloud_data = './supercloud_data/CF_gamma_deep_output' + str(y_state) + '_model_' + str(model_factor) + '_sigmoid.pth'
                    else:
                        NotImplementedError
                else:
                    if gamma_type == 'LSTM':
                        str_data = './data/CF_gamma_LSTM_output' + str(y_state) + '_model_' + str(model_factor) + '_rates_sigmoid.pth'
                        str_good_data = './good_data/data/CF_gamma_LSTM_output' + str(y_state) + '_model_' + str(model_factor) + '_rates_sigmoid.pth'
                    elif gamma_type == 'deep':
                        str_data = './data/CF_gamma_deep_output' + str(y_state) + '_model_' + str(model_factor) + '_rates_sigmoid.pth'
                        str_good_data = './good_data/data/CF_gamma_deep_output' + str(y_state) + '_model_' + str(model_factor) + '_rates_sigmoid.pth'
                    else:
                        NotImplementedError
                if i == 0:
                    if gamma_type == 'deep':
                        gamma1 = Gamma_linear_deep_nonconv_output(y_state=y_state, m_control=m_control, traj_len=traj_len, model_factor=i)
                    elif gamma_type == 'LSTM':
                        gamma1 = Gamma_linear_LSTM_output(y_state=y_state, m_control=m_control, model_factor=i)
                    else:
                        NotImplementedError
                
                    if use_good == 1:
                        try:
                            if gpu_id == -1:
                                gamma1.load_state_dict(torch.load(str_supercloud_data, map_location=torch.device('cpu')))
                            else:
                                gamma1.load_state_dict(torch.load(str_supercloud_data))
                        except:
                            try:
                                if gpu_id == -1:
                                    gamma1.load_state_dict(torch.load(str_good_data, map_location=torch.device('cpu')))
                                else:
                                    gamma1.load_state_dict(torch.load(str_good_data))
                            except:
                                if gpu_id == -1:
                                    gamma1.load_state_dict(torch.load(str_data, map_location=torch.device('cpu')))
                                else:
                                    gamma1.load_state_dict(torch.load(str_data))
                                print("No good data available")
                    else:
                        gamma1.load_state_dict(torch.load(str_data))
                    
                    gamma1.eval().to(device)
                else:
                    if gamma_type == 'deep':
                        gamma2 = Gamma_linear_deep_nonconv_output(y_state=y_state, m_control=m_control, traj_len=traj_len, model_factor=i)
                    elif gamma_type == 'LSTM':
                        gamma2 = Gamma_linear_LSTM_output(y_state=y_state, m_control=m_control, model_factor=i)
                    else:
                        NotImplementedError
                
                    if use_good == 1:
                        try:
                            if gpu_id == -1:
                                gamma2.load_state_dict(torch.load(str_supercloud_data, map_location=torch.device('cpu')))
                            else:
                                gamma2.load_state_dict(torch.load(str_supercloud_data))
                        except:
                            try:
                                if gpu_id == -1:
                                    gamma2.load_state_dict(torch.load(str_good_data, map_location=torch.device('cpu')))
                                else:
                                    gamma2.load_state_dict(torch.load(str_good_data))
                            except:
                                if gpu_id == -1:
                                    gamma2.load_state_dict(torch.load(str_data, map_location=torch.device('cpu')))
                                else:
                                    gamma2.load_state_dict(torch.load(str_data))
                                print("No good data available")
                    else:
                        gamma2.load_state_dict(torch.load(str_data))
                    
                    gamma2.eval().to(device)
            # if gamma_iter == 0 or gamma_iter == 2:
            #     model_factor = 0
            # elif gamma_iter == 1 or gamma_iter == 3:
            #     model_factor = 1
            
            # if gamma_iter == 0 or gamma_iter == 1:
            #     use_nom = 1
            # elif gamma_iter == 2 or gamma_iter == 3:
            #     use_nom = 0

            if gamma_iter == 0:
                use_nom = 1
            elif gamma_iter == 1:
                use_nom = 0

            if use_nom == 1:
                n_sample = 24000
            else:
                n_sample = 12000

            n_sample_iter = n_sample

            gamma_actual_bs = torch.ones(n_sample_iter, m_control)

            for j in range(n_sample_iter):
                temp_var = np.mod(j, 6)
                if temp_var < 4:
                    gamma_actual_bs[j, temp_var] = 0.0

            rand_ind = torch.randperm(n_sample_iter)

            gamma_actual_bs = gamma_actual_bs[rand_ind, :]

            state0 = dynamics.sample_safe(n_sample_iter // 6) + 1 * torch.randn(n_sample_iter // 6, n_state)

            state0 = state0.repeat_interleave(6, dim=0)

            state0 = state0[rand_ind, :]

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

            # print('length of failure, acc fail , acc no fail')

            new_goal = dynamics.sample_safe(1)

            new_goal = new_goal.reshape(n_state, 1)

            for k in tqdm.trange(Eval_steps):

                u_nominal = dynamics.u_nominal(state, op_point=new_goal)

                fx = dynamics._f(state, params=nominal_params)
                gx = dynamics._g(state, params=nominal_params)

                if use_nom == 0:
                    h, grad_h = cbf.V_with_jacobian(state.reshape(n_sample_iter, n_state, 1))
                    u = torch.zeros(n_sample_iter, m_control)
                    cbf_bs = 250
                    for j in range(n_sample_iter // cbf_bs):
                        batch_ind = np.arange(j * cbf_bs, (j + 1) * cbf_bs)
                        u[batch_ind] = util.fault_controller(u_nominal[batch_ind], fx[batch_ind], gx[batch_ind], h[batch_ind], grad_h[batch_ind])
                    # u = util.fault_controller(u_nominal, fx, gx, h, grad_h)
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
                    for model_iter in range(2):
                        
                        model_factor = model_iter
                        
                        if model_factor == 0:
                            gamma_NN = gamma1(state_traj[:, k - traj_len + 1:k + 1, ind_y].to(device), u_traj[:, k - traj_len + 1:k + 1, :].to(device))
                        else:
                            state_data = torch.cat((state_traj[:, k - traj_len + 1:k + 1, ind_y], state_traj_diff[:, k-traj_len + 1:k+1, ind_y]), dim=-1)
                            gamma_NN = gamma2(state_data.to(device), u_traj[:, k - traj_len + 1:k + 1, :].to(device))

                        gamma_pred = gamma_NN.reshape(n_sample_iter, m_control).clone().detach().cpu()

                        acc_ind = torch.zeros(1, m_control * 2)

                        for j in range(m_control):
                            
                            index_fault = gamma_actual_bs[:, j] < 0.5

                            index_num = torch.sum(index_fault.float())

                            if index_num > 0:
                                acc_ind[0, j] =  torch.sum((gamma_pred[index_fault, j] < 0.05).float()) / (index_num + 1e-5)
                            else:
                                acc_ind[0, j] = 1
                            
                            index_no_fault = gamma_actual_bs[:, j] > 0.5
                            
                            index_num = torch.sum(index_no_fault.float())

                            if index_num > 0:
                                acc_ind[0, j + m_control] =  torch.sum((gamma_pred[index_no_fault, j] > 0.95).float()) / (index_num + 1e-5)
                            else:
                                acc_ind[0, j + m_control] = 1
                        
                        acc_final[2 * gamma_iter + model_iter, 0, k-traj_len+1] = torch.min(acc_ind[0, 0:m_control])
                        acc_final[2 * gamma_iter + model_iter, 1, k-traj_len+1] = torch.min(acc_ind[0, m_control:])
    
        if rates == 0:
            torch.save(acc_final, './log_files/acc_output_model_' + str(model_factor) + '_'+ gamma_type + '_cbf.pt')
        else:
            torch.save(acc_final, './log_files/acc_output_model_' + str(model_factor) + '_' + gamma_type + '_rates_cbf.pt')
    else:
        print('Using previous accuracy data')

    fig = plt.figure(figsize=(20, 15))
    ax = fig.subplots(2, 1)
    ax1 = ax[0]
    ax2 = ax[1]
    if rates == 0:
        plot_name = './plots/' + 'CF_gamma_compare_output_model_' + str(model_factor)  + '_' + gamma_type + '_cbf.png'
    else:
        plot_name = './plots/' + 'CF_gamma_compare_output_model_' + str(model_factor) + '_' + gamma_type + '_rates_cbf.png'

    # ax = axes[0]

    step = np.arange(0, traj_len, 1)

    markers_on = np.arange(0, step[-1], 10)

    colors = ['b', 'r', 'b', 'r', 'm', 'y', 'k', 'w']

    for gamma_iter in range(4):
        if gamma_iter == 0:
            gamma_type = 'Model-free'
        elif gamma_iter == 1:
            gamma_type = 'Model-based'
        else:
            continue

        acc_fail = acc_final[gamma_iter, 0, :]
        acc_no_fail = acc_final[gamma_iter, 1, :]

        ax1.plot(step, acc_fail, color = colors[gamma_iter], linestyle='-', label = 'Failure: ' + gamma_type, marker="o", markevery=markers_on,markersize=20, linewidth=3)
        ax1.plot(step, acc_no_fail, color = colors[gamma_iter], linestyle='--', label = 'No Failure: ' + gamma_type, marker="^", markevery=markers_on,markersize=20, linewidth=3)
        ax1.set_ylabel('Accuracy (LQR)', fontsize = 35)
        ax1.set_xlim(step[0], step[-1])
        ax1.set_ylim(0.7, 1.01)
        ax1.tick_params(axis = "x", labelsize = 25)
        ax1.tick_params(axis = "y", labelsize = 25)
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 30)

    for gamma_iter in range(4):
        if gamma_iter == 2:
            gamma_type = 'Model-free'
        elif gamma_iter == 3:
            gamma_type = 'Model-based'
        else:
            continue

        acc_fail = acc_final[gamma_iter, 0, :]
        acc_no_fail = acc_final[gamma_iter, 1, :]

        ax2.plot(step, acc_fail, color = colors[gamma_iter], linestyle='-', label = 'Failure: ' + gamma_type, marker="o", markevery=markers_on,markersize=20, linewidth=3)
        ax2.plot(step, acc_no_fail, color = colors[gamma_iter], linestyle='--', label = 'No Failure: ' + gamma_type, marker="^", markevery=markers_on,markersize=20, linewidth=3)
        ax2.set_ylabel('Accuracy (CBF)', fontsize = 35)
        ax2.set_xlim(step[0], step[-1])
        ax2.set_ylim(0.7, 1.01)
        ax2.tick_params(axis = "x", labelsize = 25)
        ax2.tick_params(axis = "y", labelsize = 25)
        ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 30)

    plt.xlabel('Length of trajectory with failed actuator', fontsize = 35)
    
    # plt.title('Failure Test Accuracy', fontsize = 20)
    # plt.legend(fontsize=20, ncol = 1, loc='right' , bbox_to_anchor=(0.5, 1.3))
    
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
    parser.add_argument('--rates', type =int, default=1)
    parser.add_argument('--gamma_type', type=str, default='deep')
    parser.add_argument('--use_saved_data', type=int, default=1)
    args = parser.parse_args()
    main(args)
