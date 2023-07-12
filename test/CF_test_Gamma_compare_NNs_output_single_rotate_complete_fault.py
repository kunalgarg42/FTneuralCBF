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

R = torch.eye(n_state)

R[0][1] = 1.0
R[1][0] = -1.0
R[0][0] = 0.0
R[1][1] = 0.0
R[3][4] = 1.0
R[4][3] = -1.0
R[3][3] = 0.0
R[4][4] = 0.0
R[6][7] = 1.0
R[7][6] = -1.0
R[6][6] = 0.0
R[7][7] = 0.0
R[9][10] = 1.0
R[10][9] = -1.0
R[9][9] = 0.0
R[10][10] = 0.0

Rot_mat = torch.zeros(4, n_state, n_state)
Rot_mat[0, :, :] = torch.eye(n_state)

##pred3 for motor 0
Rot_mat[1,:, :] = R.T
Rot_mat[2, :, :] = torch.matmul(Rot_mat[1, :, :], R).T
Rot_mat[3, :, :] = torch.matmul(Rot_mat[2, :, :], R).T

Rot_u = torch.tensor([[0.0, 1.0, 0.0, 0.0], 
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0],
                      [1.0, 0.0, 0.0, 0.0]]).T

Rot_u_inv = torch.tensor([[0.0, 0.0, 0.0, 1.0], 
                      [1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0]]).T

##pred2 for motor 3
# Rot_mat[1,:, :] = R.T
# Rot_mat[2, :, :] = torch.matmul(Rot_mat[1, :, :], R.T)
# Rot_mat[3, :, :] = torch.matmul(Rot_mat[2, :, :], R.T)

# Rot_u = torch.tensor([[0.0, 1.0, 0.0, 0.0], 
#                       [0.0, 0.0, 1.0, 0.0],
#                       [0.0, 0.0, 0.0, 1.0],
#                       [1.0, 0.0, 0.0, 0.0]])

# Rot_u_inv = torch.tensor([[0.0, 0.0, 0.0, 1.0], 
#                       [1.0, 0.0, 0.0, 0.0],
#                       [0.0, 1.0, 0.0, 0.0],
#                       [0.0, 0.0, 1.0, 0.0]]).T
##pred1 for motor 2
# Rot_mat[1,:, :] = R.T
# Rot_mat[2, :, :] = torch.matmul(Rot_mat[1, :, :], R)
# Rot_mat[3, :, :] = torch.matmul(Rot_mat[2, :, :], R.T)

# Rot_u = torch.tensor([[0.0, 1.0, 0.0, 0.0], 
#                       [0.0, 0.0, 1.0, 0.0],
#                       [0.0, 0.0, 0.0, 1.0],
#                       [1.0, 0.0, 0.0, 0.0]]).T

# Rot_u_inv = torch.tensor([[0.0, 0.0, 0.0, 1.0], 
#                       [1.0, 0.0, 0.0, 0.0],
#                       [0.0, 1.0, 0.0, 0.0],
#                       [0.0, 0.0, 1.0, 0.0]]).T

nominal_params = config.CRAZYFLIE_PARAMS

fault = nominal_params["fault"]

use_good = 1

traj_len = 100

Eval_steps = 199

fault = nominal_params["fault"]

t = TicToc()

gpu_id = 0

gamma_type = 'deep'

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

    acc_final = torch.zeros(2, 2 * m_control, traj_len)

    use_previos_data = 0

    if use_previos_data == 1:
        try:
            acc_final = torch.load('./log_files/acc_output_model_' + str(model_factor) + '_single_complete_rotate.pt')
        except:
            use_previos_data = 0

    if use_previos_data == 0:
        print('Generating new data')

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
            n_sample = 20000
        else:
            n_sample = 1100
        
        if use_nom == 1:
            nsample_factor = 1
        else:
            nsample_factor = 1

        ind_y = torch.tensor([1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1]).bool()

        n_sample_iter = int(n_sample / nsample_factor)
    
        for gamma_iter in tqdm.trange(2):
            if gamma_iter == 0:
                gamma_type = 'LSTM'
            elif gamma_iter == 1:
                gamma_type = 'deep'
            else:
                NotImplementedError

            if gamma_type == 'LSTM':
                str_data = './data/CF_gamma_LSTM_output_single_' + str(y_state) + '_model_' + str(model_factor) + '_rates_complete_sigmoid.pth'
                str_good_data = './good_data/data/CF_gamma_LSTM_output_single_' + str(y_state) + '_model_' + str(model_factor) + '_rates_complete_sigmoid.pth'
                str_supercloud_data = './supercloud_data/CF_gamma_LSTM_output_single_' + str(y_state) + '_model_' + str(model_factor) + '_rates_complete_sigmoid.pth'
            elif gamma_type == 'deep':
                str_data = './data/CF_gamma_deep_output_single_' + str(y_state) + '_model_' + str(model_factor) + '_rates_complete_sigmoid.pth'
                str_good_data = './good_data/data/CF_gamma_deep_output_single_' + str(y_state) + '_model_' + str(model_factor) + '_rates_complete_sigmoid.pth'
                str_supercloud_data = './supercloud_data/CF_gamma_deep_output_single_' + str(y_state) + '_model_' + str(model_factor) + '_rates_complete_sigmoid.pth'
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

            gamma_actual_bs = torch.ones(n_sample_iter, m_control)

            for j in range(n_sample):
                temp_var = np.mod(j, 5)
                if temp_var < 4:
                    gamma_actual_bs[j, temp_var] = 0.0

            rand_ind = torch.randperm(n_sample_iter)

            gamma_actual_bs = gamma_actual_bs[rand_ind, :]

            state0 = dynamics.sample_safe(n_sample_iter // 5) + torch.randn(n_sample_iter // 5, n_state) * 1

            state0 = state0.repeat_interleave(5, dim=0)

            state0 = state0[rand_ind, :]

            state_traj = torch.zeros(n_sample, Eval_steps, n_state)
    
            state_traj_diff = state_traj.clone()

            u_traj = torch.zeros(n_sample, Eval_steps, m_control)

            state_traj1 = torch.zeros(n_sample, Eval_steps, n_state)
            
            state_traj_diff1 = state_traj.clone()

            u_traj1 = torch.zeros(n_sample, Eval_steps, m_control)

            state_traj2 = torch.zeros(n_sample, Eval_steps, n_state)
            
            state_traj_diff2 = state_traj.clone()

            u_traj2 = torch.zeros(n_sample, Eval_steps, m_control)

            state_traj3 = torch.zeros(n_sample, Eval_steps, n_state)
            
            state_traj_diff3 = state_traj.clone()

            u_traj3 = torch.zeros(n_sample, Eval_steps, m_control)
            
            state = state0.clone()

            state1 = torch.matmul(state, Rot_mat[1, :, :])
            state2 = torch.matmul(state, Rot_mat[2, :, :])
            state3 = torch.matmul(state, Rot_mat[3, :, :])
            
            state_no_fault = state0.clone()

            state_no_fault1 = state1.clone()

            state_no_fault2 = state2.clone()

            state_no_fault3 = state3.clone()

            for k in range(n_state):
                if k > 5:
                    state[:, k] = torch.clamp(state[:, k], sm[k] / 10, sl[k] / 10)
            
            u_nominal = dynamics.u_nominal(state)
            
            t.tic()

            new_goal = dynamics.sample_safe(1)

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

                u1 = torch.matmul(u, Rot_u)
                u2 = torch.matmul(u1, Rot_u)
                u3 = torch.matmul(u2, Rot_u)

                state_traj[:, k, :] = state.clone()
                
                state_traj_diff[:, k, :] = state_no_fault.clone() - state.clone()
                
                u_traj[:, k, :] = u.clone()

                state_traj1[:, k, :] = state1.clone()
                
                state_traj_diff1[:, k, :] = state_no_fault1.clone() - state1.clone()
                
                u_traj1[:, k, :] = u1.clone()

                state_traj1[:, k, :] = state2.clone()
                
                state_traj_diff2[:, k, :] = state_no_fault2.clone() - state2.clone()
                
                u_traj2[:, k, :] = u2.clone()

                state_traj3[:, k, :] = state3.clone()
                
                state_traj_diff3[:, k, :] = state_no_fault3.clone() - state3.clone()
                
                u_traj3[:, k, :] = u3.clone()

                gxu_no_fault = torch.matmul(gx, u.reshape(n_sample, m_control, 1))
                
                if k >= traj_len - 2:
                    u = u * gamma_actual_bs
                
                gxu = torch.matmul(gx, u.reshape(n_sample, m_control, 1))

                dx = fx.reshape(n_sample, n_state) + gxu.reshape(n_sample, n_state)

                dx_no_fault = fx.reshape(n_sample, n_state) + gxu_no_fault.reshape(n_sample, n_state)
                
                state_no_fault = state.clone() + dx_no_fault * dt

                state_no_fault1 = state1.clone() + torch.matmul(dx_no_fault, Rot_mat[1, :, :]) * dt

                state_no_fault2 = state2.clone() + torch.matmul(dx_no_fault, Rot_mat[2, :, :]) * dt

                state_no_fault3 = state3.clone() + torch.matmul(dx_no_fault, Rot_mat[3, :, :]) * dt

                state = state.clone() + dx * dt

                state1 = torch.matmul(state, Rot_mat[1, :, :])
                state2 = torch.matmul(state, Rot_mat[2, :, :])
                state3 = torch.matmul(state, Rot_mat[3, :, :])
            
                for j2 in range(n_state):
                    ind_sm = state[:, j2] > sm[j2]
                    if torch.sum(ind_sm) > 0:
                        state[ind_sm, j2] = sm[j2].repeat(torch.sum(ind_sm),)
                    ind_sl = state[:, j2] < sl[j2]
                    if torch.sum(ind_sl) > 0:
                        state[ind_sl, j2] = sl[j2].repeat(torch.sum(ind_sl),)

                if k >= traj_len - 1:
                    if model_factor == 1:
                        gamma_NN = gamma(torch.cat((state_traj[:, k - traj_len + 1:k + 1, ind_y], state_traj_diff[:, k - traj_len + 1:k + 1, ind_y]), dim=-1).to(device), u_traj[:, k - traj_len + 1:k + 1, :].to(device))
            
                        gamma_pred = gamma_NN.reshape(n_sample, m_control).clone().detach().cpu()

                        gamma_NN1 = gamma(torch.cat((state_traj1[:, k - traj_len + 1:k + 1, ind_y], state_traj_diff1[:, k - traj_len + 1:k + 1, ind_y]), dim=-1).to(device), u_traj1[:, k - traj_len + 1:k + 1, :].to(device))
                        
                        gamma_pred1 = torch.matmul(gamma_NN1.clone().detach().cpu(), Rot_u_inv).reshape(n_sample, m_control)

                        gamma_NN2 = gamma(torch.cat((state_traj2[:, k - traj_len + 1:k + 1, ind_y], state_traj_diff2[:, k - traj_len + 1:k + 1, ind_y]), dim=-1).to(device), u_traj2[:, k - traj_len + 1:k + 1, :].to(device))
                        
                        gamma_pred2 = torch.matmul(torch.matmul(gamma_NN2.clone().detach().cpu(), Rot_u_inv), Rot_u_inv).reshape(n_sample, m_control)

                        gamma_NN3 = gamma(torch.cat((state_traj3[:, k - traj_len + 1:k + 1, ind_y], state_traj_diff3[:, k - traj_len + 1:k + 1, ind_y]), dim=-1).to(device), u_traj3[:, k - traj_len + 1:k + 1, :].to(device))
                        
                        gamma_pred3 = torch.matmul(torch.matmul(torch.matmul(gamma_NN3.clone().detach().cpu(), Rot_u_inv), Rot_u_inv), Rot_u_inv).reshape(n_sample, m_control)
                    else:
                        gamma_NN = gamma(state_traj[:, k - traj_len + 1:k + 1, ind_y].to(device), u_traj[:, k - traj_len + 1:k + 1, :].to(device))
            
                        gamma_pred = gamma_NN.reshape(n_sample, m_control).clone().detach().cpu()

                        gamma_NN1 = gamma(state_traj1[:, k - traj_len + 1:k + 1, ind_y].to(device), u_traj1[:, k - traj_len + 1:k + 1, :].to(device))
                        
                        gamma_pred1 = torch.matmul(gamma_NN1.clone().detach().cpu(), Rot_u_inv).reshape(n_sample, m_control)

                        gamma_NN2 = gamma(state_traj2[:, k - traj_len + 1:k + 1, ind_y].to(device), u_traj2[:, k - traj_len + 1:k + 1, :].to(device))
                        
                        gamma_pred2 = torch.matmul(torch.matmul(gamma_NN2.clone().detach().cpu(), Rot_u_inv), Rot_u_inv).reshape(n_sample, m_control)

                        gamma_NN3 = gamma(state_traj3[:, k - traj_len + 1:k + 1, ind_y].to(device), u_traj3[:, k - traj_len + 1:k + 1, :].to(device))
                        
                        gamma_pred3 = torch.matmul(torch.matmul(torch.matmul(gamma_NN3.clone().detach().cpu(), Rot_u_inv), Rot_u_inv), Rot_u_inv).reshape(n_sample, m_control)

                    acc_ind = torch.zeros(1, m_control+1)

                    gamma_overall = torch.min(torch.cat([gamma_pred.reshape(n_sample, 1, m_control), gamma_pred1.reshape(n_sample, 1, m_control), 
                                                        gamma_pred2.reshape(n_sample, 1, m_control), gamma_pred3.reshape(n_sample, 1, m_control)], dim=1).reshape(n_sample, 4, 4), dim=1).values
                    
                    for j in range(m_control):
                        index_fault = gamma_actual_bs[:, j] == 0
                        index_num = torch.sum(index_fault.float())
                        if index_num > 0:
                            acc_ind[0, j] = 1 - torch.sum((torch.abs(gamma_actual_bs[index_fault, j] - gamma_overall[index_fault, j])).float()) / (index_num + 1e-5)
                        else:
                            acc_ind[0, j] = 1.0

                    index_no_fault = torch.sum(gamma_actual_bs, dim=1) == m_control
                    
                    index_num = torch.sum(index_no_fault == True)
                    
                    acc_ind[0, -1] = torch.sum(gamma_overall[index_no_fault, :]) / (index_num + 1e-5) / m_control

                    # print('{}, {}'.format(np.max([np.min([k - (traj_len - 2), traj_len]), 0]), acc_ind[0].numpy()))
                    
                    acc_ind = torch.zeros(1, m_control * 2)
                    acc_ind_temp = torch.zeros(4)
                    for j in range(m_control):
                        index_fault = gamma_actual_bs[:, j] < 0.5

                        index_num = torch.sum(index_fault.float())

                        if index_num > 0:
                            # acc_ind_temp[0] = torch.sum((torch.abs(gamma_pred[index_fault, j] - gamma_actual_bs[index_fault, j]) < 0.2).float()) / (index_num + 1e-5)
                            # acc_ind_temp[1] = torch.sum((torch.abs(gamma_pred1[index_fault, j] - gamma_actual_bs[index_fault, j]) < 0.2).float()) / (index_num + 1e-5)
                            # acc_ind_temp[2] = torch.sum((torch.abs(gamma_pred2[index_fault, j] - gamma_actual_bs[index_fault, j]) < 0.2).float()) / (index_num + 1e-5)
                            # acc_ind_temp[3] = torch.sum((torch.abs(gamma_pred3[index_fault, j] - gamma_actual_bs[index_fault, j]) < 0.2).float()) / (index_num + 1e-5)
                            acc_ind[0, j] =  torch.sum((torch.abs(gamma_overall[index_fault, j] - gamma_actual_bs[index_fault, j]) < 0.2).float()) / (index_num + 1e-5)
                        else:
                            acc_ind[0, j] = 1
                        
                        index_no_fault = gamma_actual_bs[:, j] > 0.5
                        
                        index_num = torch.sum(index_no_fault.float())

                        if index_num > 0:
                            # acc_ind_temp[0] = torch.sum((gamma_pred[index_no_fault, j] > 0.95).float()) / (index_num + 1e-5)
                            # acc_ind_temp[1] = torch.sum((gamma_pred1[index_no_fault, j] > 0.95).float()) / (index_num + 1e-5)
                            # acc_ind_temp[2] = torch.sum((gamma_pred2[index_no_fault, j] > 0.95).float()) / (index_num + 1e-5)
                            # acc_ind_temp[3] = torch.sum((gamma_pred3[index_no_fault, j] > 0.95).float()) / (index_num + 1e-5)
                            # acc_ind[0, j + m_control] =  torch.min(acc_ind_temp)
                            acc_ind[0, j + m_control] = torch.sum((gamma_overall[index_no_fault, j] > 0.95).float()) / (index_num + 1e-5)
                        else:
                            acc_ind[0, j + m_control] = 1
                    
                        acc_final[gamma_iter, j, k-traj_len+1] = acc_ind[0, j]
                        acc_final[gamma_iter, j + m_control, k-traj_len+1] = acc_ind[0, j + m_control]
                    # np.set_printoptions(precision=2)
                    # print(np.array(gamma_pred))
                    # print(gamma_pred[:, fault_control_index].T)
                    # print(gamma_pred1[:, fault_control_index].T)
                    # print(gamma_pred2[:, fault_control_index].T)
                    # print(gamma_pred3[:, fault_control_index].T)
                    # print(np.array(gamma_pred2))
                    # print(np.array(gamma_pred3))
                    # print(gamma_overall)
                    # print(gamma_actual_bs[:, fault_control_index].T)
                    # print(Asasas)
        torch.save(acc_final, './log_files/acc_output_model_' + str(model_factor) + '_single_complete_rotate.pt')
    else:
        print('Using previous data')

    fig = plt.figure(figsize=(28, 20))
    ax = fig.subplots(2, 2)
    
    plot_name = './plots/' + 'CF_gamma_compare_output_model_' + str(model_factor)  + '_single_complete_rotate_fault_index_' + str(fault_control_index) + '.png'

    step = np.arange(0, traj_len, 1)

    markers_on = np.arange(0, step[-1], 10)

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    for k in range(m_control):
        axk = ax[k // 2, k % 2]
        for gamma_iter in range(2):
            if gamma_iter == 0:
                gamma_type = 'LSTM'
            elif gamma_iter == 1:
                gamma_type = 'Linear MLP'

            acc_fail = acc_final[gamma_iter, k, :]
            acc_no_fail = acc_final[gamma_iter, k + m_control, :]

            axk.plot(step, acc_fail, color = colors[gamma_iter], linestyle='-', label = 'Failure: ' + gamma_type + ' actuator #'+ str(k), marker="o", markevery=markers_on,markersize=20, linewidth=3)
            axk.plot(step, acc_no_fail, color = colors[gamma_iter], linestyle='--', label = 'No Failure: ' + gamma_type + ' actuator #'+ str(k), marker="^", markevery=markers_on,markersize=20, linewidth=3)
            axk.set_xlim(step[0], step[-1])
            if model_factor == 1:
                axk.set_ylim(0.0, 1.01)
            else:
                axk.set_ylim(0.0, 1.01)
            axk.tick_params(axis = "x", labelsize = 25)
            axk.tick_params(axis = "y", labelsize = 25)

            axk.set_xlabel('Length of trajectory with failed actuator', fontsize = 30)
            axk.set_ylabel('Accuracy', fontsize = 30)
    
    # plt.title('Failure Test Accuracy', fontsize = 20)
            axk.legend(fontsize=25, ncol = 2, loc='upper center', bbox_to_anchor=(0.5, 1.3))
    
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
    parser.add_argument('--gamma_type', type =str, default='deep')
    args = parser.parse_args()
    main(args)
