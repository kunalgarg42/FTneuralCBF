import os
import sys
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import tqdm

import seaborn as sns

sns.set_theme(context="paper", font_scale=3.0, style="ticks")

sys.path.insert(1, os.path.abspath(".."))
sys.path.insert(1, os.path.abspath("."))

# from dynamics.fixed_wing_dyn import fw_dyn_ext, fw_dyn
from dynamics.Crazyflie import CrazyFlies
from trainer import config
from trainer.constraints_crazy import constraints
from trainer.datagen import Dataset_with_Grad
from trainer.trainer import Trainer
from trainer.utils import Utils
from trainer.NNfuncgrad_CF import CBF, Gamma_linear_LSTM_old

goal = torch.tensor([0.0, 0.0, 5.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

x0 = torch.tensor([[2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
dt = 0.001
n_state = 12
m_control = 4
fault = 0

FT_tol = -0.5

traj_len = config.TRAJ_LEN

nominal_params = config.CRAZYFLIE_PARAMS

fault_control_index = 1
fault_duration = config.FAULT_DURATION

fault_known = 1

n_sample = 100

def main():
    dynamics = CrazyFlies(x=x0, goal=goal, nominal_params=nominal_params, dt=dt)
    util = Utils(
        n_state=n_state,
        m_control=m_control,
        dyn=dynamics,
        params=nominal_params,
        fault=fault,
        fault_control_index=fault_control_index,
        j_const=2,
    )

    # NN_controller = NNController_new(n_state=n_state, m_control=m_control)
    NN_cbf = CBF(dynamics, n_state=n_state, m_control=m_control)
    gamma = Gamma_linear_LSTM_old(n_state=n_state, m_control=m_control, traj_len=traj_len)
    # NN_alpha = alpha_param(n_state=n_state)
    FT_cbf = CBF(dynamics, n_state=n_state, m_control=m_control)
    NN_cbf.load_state_dict(
        torch.load(
            # "./good_data/data/CF_cbf_NN_weightsCBF.pth",
            "./supercloud_data/CF_cbf_NN_weightsCBF_with_u.pth",
            map_location=torch.device("cpu"),
        )
    )
    FT_cbf.load_state_dict(
        torch.load(
            # "./good_data/data/CF_cbf_FT_weightsCBF.pth",
            "./supercloud_data/CF_cbf_FT_weightsCBF_with_u.pth",
            map_location=torch.device("cpu"),
        )
    )
    try:
        # NN_controller.load_state_dict(torch.load('./good_data/data/CF_controller_NN_weightsCBF.pth'))
        
        gamma.load_state_dict(
            torch.load(
                # "./good_data/data/CF_gamma_NN_weightssingle1.pth",
                './supercloud_data/CF_gamma_NN_class_linear_ALL_faults_no_res_LSTM.pth',
                map_location=torch.device("cpu"),
            )
        )
        # NN_alpha.load_state_dict(torch.load('./good_data/data/CF_alpha_NN_weights.pth'))
    except:
        # NN_controller.load_state_dict(torch.load('./data/CF_controller_NN_weights.pth'))
        
        gamma.load_state_dict(
            torch.load(
                "./data/CF_gamma_NN_weightssingle1.pth",
                map_location=torch.device("cpu"),
            )
        )
        # NN_alpha.load_state_dict(torch.load('./data/CF_alpha_NN_weights.pth'))

    NN_cbf.eval()

    FT_cbf.eval()

    gamma.eval()

    state = dynamics.sample_safe(n_sample)
    
    xg = dynamics.sample_safe(n_sample)

    safety_rate = 0.0
    unsafety_rate = 0.0
    h_correct = 0.0
    dot_h_correct = 0.0

    sm, sl = dynamics.state_limits()

    state_traj = torch.tensor([]).reshape(n_sample, 0, n_state)
    
    state_traj_diff = state_traj.clone()

    u_traj = torch.tensor([]).reshape(n_sample, 0, m_control)

    x_pl = np.array(state).reshape(n_sample, n_state)
    fault_activity = np.array([-1])
    actual_fault_index = np.array([-1])
    detect_activity = np.array([0]* n_sample).reshape(1, n_sample)
    NN_fault_index = np.array([-1] * n_sample).reshape(1, n_sample)

    u_pl = np.array([0] * n_sample * m_control).reshape(n_sample, m_control)
    h, _ = NN_cbf.V_with_jacobian(state.reshape(n_sample, n_state, 1))

    h_pl = np.array(h.detach()).reshape(1, n_sample)
    pred_pl = np.array([0] * n_sample).reshape(1, n_sample)

    # rand_start = random.uniform(1.01, 50)

    fault_start_epoch = config.EVAL_STEPS / 5 # 10 * math.floor(config.EVAL_STEPS / rand_start)
    fault_start = 0
    detect = np.array([0]* n_sample).reshape(1, n_sample)

    dot_h_pl = np.array([0] * n_sample).reshape(1, n_sample)

    previous_state = state.clone()
    
    fault_index_NN = np.array([-1] * n_sample).reshape(1, n_sample)

    gamma_min = torch.ones(1, n_sample)

    gamma_min_tol_ind = gamma_min > FT_tol

    gamma_min_below_tol_ind = gamma_min <= FT_tol
    
    detect_start = np.array([-1] * n_sample).reshape(1, n_sample)
    
    pred_acc = torch.zeros(1, n_sample)
    
    for i in tqdm.trange(config.EVAL_STEPS):

        u_nominal = dynamics.u_nominal(state, op_point=xg)

        u_command = u_nominal.clone()

        for j2 in range(n_state):
            ind_sm = state[:, j2] > sm[j2]
            if torch.sum(ind_sm) > 0:
                state[ind_sm, j2] = sm[j2].repeat(torch.sum(ind_sm),)
            ind_sl = state[:, j2] < sl[j2]
            if torch.sum(ind_sl) > 0:
                state[ind_sl, j2] = sl[j2].repeat(torch.sum(ind_sl),)

        fx = dynamics._f(state, params=nominal_params)
        gx = dynamics._g(state, params=nominal_params)

        h, grad_h = NN_cbf.V_with_jacobian(state.reshape(n_sample, n_state, 1))
        
        h_prev, _ = NN_cbf.V_with_jacobian(previous_state.reshape(n_sample, n_state, 1))

        u = util.fault_controller(u_nominal, fx, gx, h, grad_h)
        # u = util.neural_controller(u_nominal, fx, gx, h, grad_h, detect)

        u = u.clone().type(torch.float32)

        u = u.reshape(n_sample, m_control)

        u_command = u.clone()

        gxu_no_fault = torch.matmul(gx, u.reshape(n_sample, m_control, 1))

        dx_no_fault = fx.reshape(n_sample, n_state) + gxu_no_fault.reshape(n_sample, n_state)

        if fault_start_epoch <= i <= fault_start_epoch + fault_duration:
            u[:, fault_control_index] = torch.zeros(n_sample, )
            fault_start = 1.0
        else:
            fault_start = 0.0

        u = u.clone().type(torch.float32)
        
        gxu = torch.matmul(gx, u.reshape(n_sample, m_control, 1))

        dx = fx.reshape(n_sample, n_state) + gxu.reshape(n_sample, n_state)

        dot_h = (h - h_prev) / dt + 0.01 * h
        
        if i >= fault_start_epoch + traj_len:
            gamma_NN = gamma(state_traj.reshape(n_sample, traj_len, n_state), u_traj.reshape(n_sample, traj_len, m_control))
            
            gamma_NN = gamma_NN.detach()

            gamma_min = torch.min(gamma_NN, dim=1)

            fault_index_NN = torch.argmin(gamma_NN, dim=1).numpy().reshape(1, n_sample)
            
            # if fault_start == 1:
            #     pred_acc = 1 - torch.abs(gamma_NN[:, fault_control_index])
            # else:
            #     pred_acc = 1 - torch.abs(gamma_NN[:, fault_control_index] - 1)
            pred_acc = 0.0
            for control_iter in range(m_control):
                if fault_start == 1:
                    if control_iter == fault_control_index:
                        pred_acc += (gamma_NN[:, control_iter] < 0).float()
                    else:
                        pred_acc += (gamma_NN[:, control_iter] > 0).float()
                else:
                    pred_acc += torch.sum((gamma_NN[:, control_iter] > 0).float())

            pred_acc = pred_acc / m_control

            gamma_min_tol_ind = gamma_min.values > FT_tol
            gamma_min_below_tol_ind = gamma_min.values <= FT_tol

            gamma_min = gamma_min.values

            # if gamma_min >= FT_tol:

            # print(torch.sum(pred_acc) / n_sample)
            # print(gamma_min)
            # gamma_NN[gamma_min_tol_ind, fault_index_NN] = torch.ones(torch.sum(gamma_min_tol_ind),)
            # gamma_min[gamma_min_tol_ind] = torch.ones(torch.sum(gamma_min_tol_ind),)

        else:
            gamma_min = torch.ones(1, n_sample)
        
        num_gamma_min_tol_ind = int(torch.sum(gamma_min_tol_ind))
        if gamma_min_tol_ind[0].shape == torch.Size([]):
            gamma_min_tol_ind = gamma_min_tol_ind.reshape(1, gamma_min_tol_ind.shape[0])

        # gamma_min_tol_ind = gamma_min_tol_ind.reshape(1, num_gamma_min_tol_ind)
        if num_gamma_min_tol_ind > 0:
            # 
            detect[0, gamma_min_tol_ind[0]] = np.array([0] * num_gamma_min_tol_ind).reshape(1, num_gamma_min_tol_ind)
            # else:
            # print(gamma_min_tol_ind[0].shape)
                # detect[0, gamma_min_tol_ind[0]] = np.array([0] * num_gamma_min_tol_ind).reshape(1, num_gamma_min_tol_ind) # torch.zeros(1, torch.sum(gamma_min_tol_ind))        
        
        fault_index_NN[0, gamma_min_tol_ind[0]] = -1 * np.array([1] * num_gamma_min_tol_ind).reshape(1, num_gamma_min_tol_ind)

        detect_start_ind = detect_start < 0

        detect_start_ind = np.array(detect_start_ind) * np.array(gamma_min_below_tol_ind)
        
        num_detect_start_ind = int(sum(detect_start_ind[0]))

        detect_start[0, detect_start_ind[0]] = np.array([i] * num_detect_start_ind).reshape(1, num_detect_start_ind)

        detect_start_ind = i - detect_start > 50

        # if gamma_min > FT_tol:
        #     detect = 0
        #     fault_index_NN = -1
        # else:
        #     if detect_start == -1:
        #         detect_start = i

        # if (gamma_min < FT_tol and i - detect_start > 50): # or detect == 1:
        num_detect_start = int(sum(detect_start_ind[0]))

        detect[0, detect_start_ind[0]] = np.array([1] * num_detect_start).reshape(1, num_detect_start)

        h, grad_h = FT_cbf.V_with_jacobian(state.reshape(n_sample, n_state, 1))

        u_new = util.fault_controller(u_nominal, fx, gx, h, grad_h)

        # u_new = util.neural_controller(u_nominal, fx, gx, h, grad_h, detect)
        u_new = u_new.clone().type(torch.float32).reshape(n_sample, m_control)
        
        u_new[:, fault_control_index] = u[:, fault_control_index].clone()
        
        u[detect_start_ind[0], :] = u_new[detect_start_ind[0], :].clone()

        u_command = u.clone()

        gxu_no_fault = torch.matmul(gx, u.reshape(n_sample, m_control, 1))

        dx_no_fault = fx.reshape(n_sample, n_state) + gxu_no_fault.reshape(n_sample, n_state)

        if fault_start_epoch <= i <= fault_start_epoch + fault_duration:
            u[:, fault_control_index] = torch.zeros(n_sample, ) #  0 * (torch.sin(torch.tensor(i / 100)) ** 2) * um[0, 0].clone() #  torch.rand(1) / 4
            fault_start = 1.0
        else:
            fault_start = 0.0

        gxu = torch.matmul(gx, u.reshape(n_sample, m_control, 1))

        dx = fx.reshape(n_sample, n_state) + gxu.reshape(n_sample, n_state)

        detect_activity = np.vstack((detect_activity, detect))
        
        if fault_start == 1:
            actual_fault_index = np.vstack((actual_fault_index, fault_control_index))
        else:
            actual_fault_index = np.vstack((actual_fault_index, -1.0))

        # if detect == 1:
        NN_fault_index = np.vstack((NN_fault_index, fault_index_NN))
        # else:
            # NN_fault_index = np.vstack((NN_fault_index, -1.0))

        # if fault_known == 0:
        #     dot_h_pl = np.vstack((dot_h_pl, dot_h.clone().detach().numpy()))
        
        # dot_h = util.doth_max_alpha(h, grad_h, fx, gx, um, ul)

        # if fault_known == 1:
        dot_h_pl = np.vstack((dot_h_pl, dot_h.clone().detach().numpy().reshape(1, n_sample)))
        
        state_next = state + dx * dt

        state_next_no_fault = state + dx_no_fault * dt

        previous_state = state.clone()

        is_safe = int(util.is_safe(state)[0])

        is_unsafe = int(util.is_unsafe(state)[0])
        
        safety_rate += is_safe / config.EVAL_STEPS / n_sample

        unsafety_rate += is_unsafe / config.EVAL_STEPS / n_sample
        
        h_correct += (
            is_safe * int((h >= 0)[0]) / config.EVAL_STEPS / n_sample
            + is_unsafe * int((h < 0)[0]) / config.EVAL_STEPS / n_sample
        )

        dot_h_correct += torch.sum(torch.sign(dot_h.clone().detach()) >= 0) / config.EVAL_STEPS / n_sample

        x_pl = np.vstack((x_pl, np.array(state.clone().detach()).reshape(n_sample, n_state)))

        fault_activity = np.vstack((fault_activity, fault_start))
        
        u_pl = np.vstack((u_pl, np.array(u.clone().detach()).reshape(n_sample, m_control)))
        h_pl = np.vstack((h_pl, np.array(h.clone().detach()).reshape(1, n_sample)))

        if i >= traj_len:
            pred_pl = np.vstack((pred_pl, pred_acc))
        else:
            pred_pl = np.vstack((pred_pl, np.array([0.0] * n_sample).reshape(1, n_sample)))

        state = state_next.clone()

        state_diff = state_next_no_fault - state
        
        state_traj = torch.cat([state_traj, state_next.reshape(n_sample, 1, n_state)], dim=-2)
        state_traj = state_traj[:, -traj_len:, :]

        state_traj_diff = torch.cat([state_traj_diff, state_diff.reshape(n_sample, 1, n_state)], dim=-2)
        state_traj_diff = state_traj_diff[:, -traj_len:, :]

        u_traj = torch.cat([u_traj, u_command.reshape(n_sample, 1, m_control)], dim=-2)
        u_traj = u_traj[:, -traj_len:, :]

        
    print(safety_rate)
    print(unsafety_rate)
    print(h_correct)
    print(dot_h_correct)
    

    
if __name__ == "__main__":
    main()
