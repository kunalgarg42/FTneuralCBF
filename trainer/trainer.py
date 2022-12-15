import torch
from torch import nn
import numpy as np
from pytictoc import TicToc

torch.autograd.set_detect_anomaly(True)

t = TicToc()


class Trainer(object):

    def __init__(self,
                 cbf,
                 dataset,
                 dyn,
                 params,
                 n_state,
                 m_control,
                 gamma=None,
                 num_traj=1,
                 j_const=1,
                 dt=0.05,
                 action_loss_weight=0.1,
                 gpu_id=-1,
                 lr_decay_stepsize=-1,
                 fault=0,
                 fault_control_index=-1):

        self.params = params
        self.n_state = n_state
        self.m_control = m_control
        self.j_const = j_const
        self.dyn = dyn
        self.cbf = cbf
        self.gamma = gamma
        self.dataset = dataset
        self.fault = fault
        self.fault_control_index = fault_control_index
        self.num_traj = num_traj
        self.cbf_optimizer = torch.optim.Adam(
            self.cbf.parameters(), lr=1e-5, weight_decay=1e-5)
        if gamma is not None:
            self.gamma_optimizer = torch.optim.Adam(
                self.gamma.parameters(), lr=1e-4, weight_decay=1e-5)
        # # self.controller_optimizer = FxTS_Momentum(
        #     self.controller.parameters(), lr=1e-5, momentum=0.2)
        # self.cbf_optimizer = FxTS_Momentum(
        #     self.cbf.parameters(), lr=5e-5, momentum=0.2)
        # self.alpha_optimizer = FxTS_Momentum(
        #     self.alpha.parameters(), lr=1e-5, momentum=0.2)

        self.dt = dt

        self.action_loss_weight = action_loss_weight
        # if gpu_id >=0, use gpu in training
        self.gpu_id = gpu_id

        # the learning rate is decayed when self.train_cbf_and_controller is called
        # lr_decay_stepsize times
        self.lr_decay_stepsize = lr_decay_stepsize
        if lr_decay_stepsize >= 0:
            self.cbf_lr_scheduler = torch.optim.lr_scheduler.StepLR(
                self.cbf_optimizer, step_size=lr_decay_stepsize, gamma=0.5)

    # noinspection PyProtectedMember,PyUnboundLocalVariable
    def train_cbf_and_controller(self, iter_NN=0, eps=0.1, eps_deriv=0.03, train_CF=0):
        batch_size = 4000 + int(iter_NN / 4) * 2000
        opt_iter = int(self.dataset.n_pts / batch_size)
        loss_np = 0.0
        loss_h_safe_np = 0.0
        loss_h_dang_np = 0.0
        loss_deriv_safe_np = 0.0
        loss_deriv_mid_np = 0.0
        loss_deriv_dang_np = 0.0
        loss_alpha_np = 0.0
        loss_action_np = 0.0

        acc_np = np.zeros((5,), dtype=np.float32)
        # print("training both CBF and u")
        # t.tic()
        u_nominal = 0.1 * torch.ones(batch_size, self.m_control)
        dang_loss = 1
        if self.fault == 1:
            um, ul = self.dyn.control_limits()
            um = um.reshape(1, self.m_control).repeat(batch_size, 1)
            ul = ul.reshape(1, self.m_control).repeat(batch_size, 1)

            um = um.type(torch.FloatTensor)
            ul = ul.type(torch.FloatTensor)
            if self.gpu_id >= 0:
                um = um.cuda(self.gpu_id)
                ul = ul.cuda(self.gpu_id)

        for j in range(10):
            # if j<5:
            #     deriv_factor = 0
            # else:
            deriv_factor = 1
            for i in range(opt_iter):
                # t.tic()
                # print(i)
                state, _, _ = self.dataset.sample_data(batch_size, i)

                if self.gpu_id >= 0:
                    state = state.cuda(self.gpu_id)
                    u_nominal = u_nominal.cuda(self.gpu_id)
                    self.cbf.to(torch.device(self.gpu_id))
                    self.controller.to(torch.device(self.gpu_id))
                    self.alpha.to(torch.device(self.gpu_id))

                safe_mask, dang_mask, mid_mask = self.get_mask(state)

                u = self.controller(state, u_nominal)
                h, grad_h = self.cbf.V_with_jacobian(state)
                alpha = self.alpha(state)

                dsdt = self.nominal_dynamics(state, u.reshape(batch_size, self.m_control, 1))

                dsdt = torch.reshape(dsdt, (batch_size, self.n_state))

                dot_h = torch.matmul(grad_h.reshape(batch_size, 1, self.n_state),
                                     dsdt.reshape(batch_size, self.n_state, 1))
                dot_h = dot_h.reshape(batch_size, 1)

                if self.fault == 0:
                    deriv_cond = dot_h + alpha * h
                else:
                    gx = self.dyn._g(state, self.params)
                    LhG = torch.matmul(grad_h, gx)
                    sign_grad_h = torch.sign(LhG).reshape(batch_size, 1, self.m_control)
                    doth = - sign_grad_h[:, 0, self.fault_control_index].reshape(batch_size, 1) * \
                           um[:, self.fault_control_index].reshape(batch_size, 1) - \
                           (1 - sign_grad_h[:, 0, self.fault_control_index].reshape(batch_size, 1)) * \
                           ul[:, self.fault_control_index].reshape(batch_size, 1)

                    deriv_cond = dot_h + alpha * h + doth.reshape(batch_size, 1)

                num_safe = torch.sum(safe_mask)
                num_dang = torch.sum(dang_mask)
                num_mid = torch.sum(mid_mask)

                loss_h_safe = torch.sum(
                    nn.ReLU()(eps - h).reshape(1, batch_size) * safe_mask.reshape(1, batch_size)) / (1e-5 + num_safe)
                loss_h_dang = dang_loss * torch.sum(
                    nn.ReLU()(h + eps).reshape(1, batch_size) * dang_mask.reshape(1, batch_size)) / (1e-5 + num_dang)

                acc_h_safe = torch.sum(
                    (h >= 0).reshape(1, batch_size).float() * safe_mask.reshape(1, batch_size)) / (1e-5 + num_safe)
                acc_h_dang = torch.sum(
                    (h < 0).reshape(1, batch_size).float() * dang_mask.reshape(1, batch_size)) / (1e-5 + num_dang)

                loss_deriv_safe = deriv_factor * torch.sum(
                    nn.ReLU()(eps_deriv - deriv_cond).reshape(1, batch_size) * safe_mask.reshape(1, batch_size)) / (
                                          1e-5 + num_safe)

                loss_deriv_mid = deriv_factor * 0.1 * torch.sum(
                    nn.ReLU()(eps_deriv - deriv_cond).reshape(1, batch_size) * mid_mask.reshape(1, batch_size)) / (
                                         1e-5 + num_mid)
                
                loss_alpha = 0 * loss_h_safe.clone()
                
                loss_deriv_dang = deriv_factor * 0.01 * torch.sum(
                    nn.ReLU()(eps_deriv - deriv_cond).reshape(1, batch_size) * dang_mask.reshape(1, batch_size)) / (
                                          1e-5 + num_dang)
                if train_CF == 1:
                    if self.fault == 0:
                        loss_action = 10 * torch.sum(nn.ReLU()(0.07 - u)) / batch_size
                    else:
                        loss_action = 0.0
                        for cont_ind in range(self.m_control):
                            if cont_ind != self.fault_control_index:
                                loss_action += 10 * torch.sum(nn.ReLU()(0.07 - u[:, cont_ind])) / batch_size
                else:
                    loss_action = 0 * loss_alpha

                acc_deriv_safe = torch.sum(
                    (deriv_cond > 0).reshape(1, batch_size).float() * safe_mask) / (1e-5 + num_safe)
                acc_deriv_dang = torch.sum(
                    (deriv_cond > 0).reshape(1, batch_size).float() * dang_mask) / (1e-5 + num_dang)
                acc_deriv_mid = torch.sum((deriv_cond > 0).reshape(1, batch_size).float() * mid_mask) / (1e-5 + num_mid)

                loss = loss_h_safe + loss_h_dang + loss_deriv_safe + loss_deriv_dang + loss_deriv_mid + loss_action


                self.controller_optimizer.zero_grad()
                self.cbf_optimizer.zero_grad()
                self.alpha_optimizer.zero_grad()

                loss.backward()

                # P_grad = 0
                # for p in self.controller.parameters():
                #     P_grad += torch.sum(torch.linalg.norm(p.grad))
                #
                # # print(self.controller.parameters())
                # print(P_grad)

                # P_grad = 0
                # for p in self.alpha.parameters():
                #     P_grad += torch.sum(torch.linalg.norm(p.grad))
                #
                # print(P_grad)

                self.controller_optimizer.step()
                self.cbf_optimizer.step()
                self.alpha_optimizer.step()

                # log statics
                acc_np[0] += acc_h_safe.detach().cpu()
                acc_np[1] += acc_h_dang.detach().cpu()

                acc_np[2] += acc_deriv_safe.detach()
                acc_np[3] += acc_deriv_dang.detach()
                acc_np[4] += acc_deriv_mid.detach()

                loss_np += loss.detach().cpu().numpy()
                loss_h_safe_np += loss_h_safe.detach().cpu().numpy()
                loss_h_dang_np += loss_h_dang.detach().cpu().numpy()
                loss_deriv_safe_np += loss_deriv_safe.detach().cpu().numpy()
                loss_deriv_mid_np += loss_deriv_mid.detach().cpu().numpy()
                loss_deriv_dang_np += loss_deriv_dang.detach().cpu().numpy()
                loss_alpha_np += loss_alpha.detach().cpu().numpy()
                loss_action_np += loss_action.detach().cpu().numpy()
                # print("reached here")

        acc_np /= opt_iter * 10
        loss_np /= opt_iter * 10
        loss_h_safe_np /= opt_iter * 10
        loss_h_dang_np /= opt_iter * 10
        loss_deriv_safe_np /= opt_iter * 10
        loss_deriv_mid_np /= opt_iter * 10
        loss_deriv_dang_np /= opt_iter * 10
        loss_alpha_np /= opt_iter * 10
        loss_action_np /= opt_iter * 10

        if self.lr_decay_stepsize >= 0:
            # learning rate decay
            self.cbf_lr_scheduler.step()
            # self.controller_lr_scheduler.step()

        return loss_np, acc_np, loss_h_safe_np, loss_h_dang_np, loss_alpha_np, loss_deriv_safe_np, loss_deriv_dang_np, loss_deriv_mid_np, loss_action_np

    def train_cbf(self, batch_size=5000, opt_iter=20, eps=0.1, eps_deriv=0.03):
        loss_np = 0.0
        loss_h_safe_np = 0.0
        loss_h_dang_np = 0.0
        loss_deriv_safe_np = 0.0
        loss_deriv_mid_np = 0.0
        loss_deriv_dang_np = 0.0
        loss_alpha_np = 0.0
        acc_np = np.zeros((5,), dtype=np.float32)
        # print("training only CBF")
        # t.tic()
        um, ul = self.dyn.control_limits()
        um = um.reshape(1, self.m_control).repeat(batch_size, 1)
        ul = ul.reshape(1, self.m_control).repeat(batch_size, 1)

        um = um.type(torch.FloatTensor)
        ul = ul.type(torch.FloatTensor)
        if self.gpu_id >= 0:
            um = um.cuda(self.gpu_id)
            ul = ul.cuda(self.gpu_id)
        
        opt_count = 20
        for _ in range(opt_count):
            for i in range(opt_iter):
                # t.tic()
                # print(i)
                state, _, _ = self.dataset.sample_data(batch_size, i)
                if self.gpu_id >= 0:
                    state = state.cuda(self.gpu_id)
                    self.cbf.to(torch.device(self.gpu_id))

                safe_mask, dang_mask, mid_mask = self.get_mask(state)

                h, grad_h = self.cbf.V_with_jacobian(state)

                dot_h_max = self.doth_max(h, state, grad_h, um, ul)
                deriv_cond = dot_h_max  # + alpha.reshape(1, batch_size) * h.reshape(1, batch_size)

                num_safe = torch.sum(safe_mask)
                num_dang = torch.sum(dang_mask)
                num_mid = torch.sum(mid_mask)

                acc_h_safe = torch.sum(
                    (h >= 0).reshape(1, batch_size).float() * safe_mask.reshape(1, batch_size)) / (1e-5 + num_safe)
                acc_h_dang = torch.sum(
                    (h < 0).reshape(1, batch_size).float() * dang_mask.reshape(1, batch_size)) / (1e-5 + num_dang)

                loss_h_safe = 10 * torch.sum(
                    nn.ReLU()(eps - h).reshape(1, batch_size) * safe_mask.reshape(1, batch_size)) / (
                                      1e-5 + num_safe) / (acc_h_safe.clone().detach() + 1e-5)

                loss_h_dang = 10 * torch.sum(
                    nn.ReLU()(h + eps).reshape(1, batch_size) * dang_mask.reshape(1, batch_size)) / (
                                      1e-5 + num_dang) / (acc_h_dang.clone().detach() + 1e-5)

                acc_deriv_safe = torch.sum((deriv_cond > 0).float() * safe_mask) / (1e-5 + num_safe)
                acc_deriv_dang = torch.sum((deriv_cond > 0).float() * dang_mask) / (1e-5 + num_dang)
                acc_deriv_mid = torch.sum((deriv_cond > 0).float() * mid_mask) / (1e-5 + num_mid)

                loss_deriv_safe = torch.sum(
                    nn.ReLU()(eps_deriv - deriv_cond).reshape(1, batch_size) * safe_mask.reshape(1, batch_size)) / (
                                          1e-5 + num_safe) / (acc_deriv_safe.detach() + 1e-5)
                loss_deriv_dang = torch.sum(
                    nn.ReLU()(eps_deriv - deriv_cond).reshape(1, batch_size) * dang_mask.reshape(1, batch_size)) / (
                                          1e-5 + num_dang) / (acc_deriv_dang.detach() + 1e-5)
                loss_deriv_mid = torch.sum(
                    nn.ReLU()(eps_deriv - deriv_cond).reshape(1, batch_size) * mid_mask.reshape(1, batch_size)) / (
                                         1e-5 + num_mid) / (acc_deriv_mid.detach() + 1e-5)

                loss = loss_h_safe + loss_h_dang + loss_deriv_safe + loss_deriv_dang + loss_deriv_mid

                self.cbf_optimizer.zero_grad()
                # self.alpha_optimizer.zero_grad()

                loss.backward(retain_graph=True)

                self.cbf_optimizer.step()
                # self.alpha_optimizer.step()

                # log statics
                acc_np[0] += acc_h_safe.detach().cpu()
                acc_np[1] += acc_h_dang.detach().cpu()

                acc_np[2] += acc_deriv_safe.detach()
                acc_np[3] += acc_deriv_dang.detach()
                acc_np[4] += acc_deriv_mid.detach()

                loss_np += loss.detach().cpu().numpy()
                loss_h_safe_np += loss_h_safe.detach().cpu().numpy()
                loss_h_dang_np += loss_h_dang.detach().cpu().numpy()
                loss_deriv_safe_np += loss_deriv_safe.detach().cpu().numpy()
                loss_deriv_mid_np += loss_deriv_mid.detach().cpu().numpy()
                loss_deriv_dang_np += loss_deriv_dang.detach().cpu().numpy()

        acc_np /= opt_iter * opt_count
        loss_np /= opt_iter * opt_count
        loss_h_safe_np /= opt_iter * opt_count
        loss_h_dang_np /= opt_iter * opt_count
        loss_deriv_safe_np /= opt_iter * opt_count
        loss_deriv_mid_np /= opt_iter * opt_count
        loss_deriv_dang_np /= opt_iter * opt_count
        loss_alpha_np /= opt_iter * opt_count

        if self.lr_decay_stepsize >= 0:
            # learning rate decay
            self.cbf_lr_scheduler.step()
            # self.controller_lr_scheduler.step()

        return loss_np, acc_np, loss_h_safe_np, loss_h_dang_np, loss_deriv_safe_np, loss_deriv_dang_np, loss_deriv_mid_np

    def train_gamma(self, traj_len, batch_size=5000, opt_iter=10, eps=0.1, eps_deriv=0.01):
        loss_np = 0.0

        if batch_size > self.dataset.n_pts:
            batch_size = self.dataset.n_pts
        
        ns = int(batch_size / traj_len)
        opt_count = 10
        for _ in range(opt_count):
            for i in range(opt_iter):
                # t.tic()
                # print(i)
                state, u, gamma_actual = self.dataset.sample_data_all(batch_size, ns, i)
                
                if self.gpu_id >= 0:
                    state = state.cuda(self.gpu_id)
                    u = u.cuda(self.gpu_id)
                    gamma_actual = gamma_actual.cuda(self.gpu_id)
                    # gamma_data = gamma_data.cuda(self.gpu_id)
                    self.gamma.to(torch.device(self.gpu_id))

                gamma_data = self.gamma_gen(state, u, traj_len)
                
                num_gamma = gamma_data.shape[0]

                gamma_error = gamma_data - gamma_actual

                gamma_error = torch.abs(gamma_error) * 100

                loss_1 = torch.sum(nn.ReLU()(-eps_deriv + gamma_error[:, 0]).reshape(1, num_gamma)) / num_gamma

                loss_2 = torch.sum(nn.ReLU()(-eps_deriv + gamma_error[:, 1]).reshape(1, num_gamma)) / num_gamma

                loss_3 = torch.sum(nn.ReLU()(-eps_deriv + gamma_error[:, 2]).reshape(1, num_gamma)) / num_gamma

                loss_4 = torch.sum(nn.ReLU()(-eps_deriv + gamma_error[:, 3]).reshape(1, num_gamma)) / num_gamma

                loss = loss_1 + loss_2 + loss_3 + loss_4

                self.gamma_optimizer.zero_grad()
                # self.alpha_optimizer.zero_grad()

                loss.backward()

                self.gamma_optimizer.step()
                # self.alpha_optimizer.step()

                # log statics
                loss_np += loss.detach().cpu().numpy()
                
        loss_np /= opt_iter * opt_count

        return loss_np


    def doth_max(self, h, state, grad_h, um, ul):
        bs = grad_h.shape[0]

        # LhG = LhG.detach().cpu()
        fx = self.dyn._f(state, self.params)
        gx = self.dyn._g(state, self.params)
        vec_ones = 10 * torch.ones(bs, 1)
        if self.gpu_id >= 0:
            fx = fx.cuda(self.gpu_id)
            gx = gx.cuda(self.gpu_id)
            vec_ones = vec_ones.cuda(self.gpu_id)

        doth = torch.matmul(grad_h, fx)

        LhG = torch.matmul(grad_h, gx).reshape(bs, self.m_control)
        LhG = torch.hstack((LhG, h))

        sign_grad_h = torch.sign(LhG).reshape(bs, 1, self.m_control + 1)

        um = torch.hstack((um, vec_ones)).reshape(self.m_control + 1, bs)
        ul = torch.hstack((ul, -1 * vec_ones)).reshape(self.m_control + 1, bs)

        uin = um.reshape(self.m_control + 1, bs) * \
              (sign_grad_h > 0).reshape(self.m_control + 1, bs) - ul.reshape(self.m_control + 1, bs) * (
                      sign_grad_h <= 0).reshape(self.m_control + 1, bs)

        # if self.fault == 0:
        doth = doth + torch.matmul(torch.abs(LhG).reshape(bs, 1, self.m_control + 1),
                                   uin.reshape(bs, self.m_control + 1, 1))
        if self.fault == 1:
            doth = doth.reshape(bs, 1) - torch.abs(LhG[:, self.fault_control_index]).reshape(bs, 1) * uin[self.fault_control_index,:].reshape(
                bs, 1)

            # ran_tensor = torch.randn(bs, 1).to(state.get_device())
            # ran_tensor = (ran_tensor > 0).int()

            doth = doth.reshape(bs, 1) - torch.abs(LhG[:, self.fault_control_index]).reshape(bs, 1) * um[self.fault_control_index,:].reshape(
                bs, 1)
        # else:
        #     for i in range(self.m_control + 1):
        #         if i == self.fault_control_index:
        #             doth = doth.reshape(bs, 1) + torch.abs(LhG[:, i]).reshape(bs, 1) * uin[i, :].reshape(bs, 1)
        #         else:
        #             doth = doth.reshape(bs, 1) + torch.abs(LhG[:, i]).reshape(bs, 1) * uin[i, :].reshape(bs, 1)

        return doth.reshape(1, bs)

    def get_mask(self, state):
        """
        args:
            state (bs, n_state)
        returns:
            safe_mask (bs, k_obstacle)
            mid_mask  (bs, k_obstacle)
            dang_mask (bs, k_obstacle)
        """
        safe_mask = self.dyn.safe_mask(state, self.fault).float()
        dang_mask = self.dyn.unsafe_mask(state).float()
        mid_mask = (1 - safe_mask) * (1 - dang_mask)

        return safe_mask, dang_mask, mid_mask

    def nominal_dynamics(self, state, u):
        """
        args:
            state (n_state,)
            u (m_control,)
        returns:
            dsdt (n_state,)
        """

        fx = self.dyn._f(state, self.params)
        gx = self.dyn._g(state, self.params)

        # for j in range(self.m_control):
        if self.fault == 1:
            u[:, self.fault_control_index] = 0 * u[:, 0].clone()
        #     # else:
        #     #     u[:, j] = u[:, j].clone().detach().requires_grad_(True).reshape(batch_size, 1)

        dsdt = fx + torch.matmul(gx, u)

        return dsdt

    def gamma_gen(self, state, u, traj_len):
        """
        args:
            state (n_state,)
            u (m_control,)
        returns:
            gamma (m_control,)
        """

        ns = int(state.shape[0] / traj_len)

        gamma_data = torch.zeros(ns, self.m_control)

        if state.get_device() >= 0:
            state = state.cuda(self.gpu_id)
            u = u.cuda(self.gpu_id)
            gamma_data = gamma_data.cuda(self.gpu_id)
            self.gamma.to(torch.device(self.gpu_id))

        state = state.reshape(ns, traj_len, self.n_state)
        u = u.reshape(ns, traj_len, self.m_control)
        # for i in range(self.num_traj):
        gamma_dot = self.gamma(state, u)
        gamma_dot = gamma_dot.reshape(ns, self.m_control)
        # gamma_temp = torch.sum(gamma_dot, dim=0)
        # gamma_temp = gamma_temp.reshape(ns, self.m_control)
        # for j in range(traj_len-1):
        #     gamma_temp = gamma_temp + self.gamma(state[j + ns*j:j + ns * (j+1), :].reshape(ns, self.n_state), u[j + ns*j:j + ns * (j+1), :].reshape(ns, self.m_control)) * self.dt

        gamma_data = gamma_dot.clone()
        # gamma_temp = 0 * gamma_temp.clone()

        return gamma_data