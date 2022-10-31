import torch
import math
import scipy
from torch import nn
import numpy as np
from qpsolvers import solve_qp
from scipy.sparse import identity
from scipy.sparse import csc_matrix

from pytictoc import TicToc

torch.autograd.set_detect_anomaly(True)

t = TicToc()


class Trainer(object):

    def __init__(self,
                 controller,
                 cbf,
                 alpha,
                 dataset,
                 dyn,
                 n_pos,
                 params,
                 n_state,
                 m_control,
                 j_const=1,
                 dt=0.05,
                 safe_alpha=0.3,
                 dang_alpha=0.4,
                 action_loss_weight=0.0001,
                 gpu_id=-1,
                 lr_decay_stepsize=-1,
                 fault=0,
                 fault_control_index=-1):

        self.params = params
        self.n_state = n_state
        self.m_control = m_control
        self.j_const = j_const
        self.controller = controller
        self.dyn = dyn
        self.cbf = cbf
        self.alpha = alpha
        self.dataset = dataset
        self.fault = fault
        self.fault_control_index = fault_control_index

        self.controller_optimizer = torch.optim.Adam(
            self.controller.parameters(), lr=5e-4, weight_decay=1e-5)
        self.cbf_optimizer = torch.optim.Adam(
            self.cbf.parameters(), lr=1e-4, weight_decay=1e-5)
        self.alpha_optimizer = torch.optim.Adam(
            self.alpha.parameters(), lr=1e-4, weight_decay=1e-5)
        # self.controller_optimizer = FxTS_Momentum(
        #     self.controller.parameters(), lr=5e-4,momentum = 0.2)
        # self.cbf_optimizer = FxTS_Momentum(
        #     self.cbf.parameters(), lr=1e-4,momentum = 0.2)
        # self.alpha_optimizer = FxTS_Momentum(
        #     self.alpha.parameters(), lr=5e-4,momentum = 0.2)

        self.n_pos = n_pos
        self.dt = dt
        self.safe_alpha = safe_alpha
        self.dang_alpha = dang_alpha
        self.action_loss_weight = action_loss_weight
        # if gpu_id >=0, use gpu in training
        self.gpu_id = gpu_id

        # the learning rate is decayed when self.train_cbf_and_controller is called
        # lr_decay_stepsize times
        self.lr_decay_stepsize = lr_decay_stepsize
        if lr_decay_stepsize >= 0:
            self.cbf_lr_scheduler = torch.optim.lr_scheduler.StepLR(
                self.cbf_optimizer, step_size=lr_decay_stepsize, gamma=0.5)
            self.controller_lr_scheduler = torch.optim.lr_scheduler.StepLR(
                self.controller_optimizer, step_size=lr_decay_stepsize, gamma=0.5)

    def train_cbf_and_controller(self, batch_size=1000, opt_iter=100, eps=0.1, eps_deriv=0.03, eps_action=0.2):
        loss_np = 0.0
        loss_h_safe_np = 0.0
        loss_h_dang_np = 0.0
        loss_deriv_safe_np = 0.0
        loss_deriv_mid_np = 0.0
        loss_deriv_dang_np = 0.0
        loss_alpha_np = 0.0
        loss_action_np = 0.0
        acc_np = np.zeros((5,), dtype=np.float32)
        print("training")
        for j in range(10):
            for i in range(opt_iter):
                # t.tic()
                state, u, u_nominal = self.dataset.sample_data(batch_size, i)
                u_nominal = torch.from_numpy(u_nominal)

                if self.gpu_id >= 0:
                    state = state.cuda(self.gpu_id)
                    u = u.cuda(self.gpu_id)
                    u_nominal = u_nominal.cuda(self.gpu_id)

                if self.fault == 1:
                    u[:, self.fault_control_index] = u[:, self.fault_control_index].detach()

                safe_mask, dang_mask, mid_mask = self.get_mask(state)

                h, grad_h = self.cbf.V_with_jacobian(state)

                u = self.controller(state, u_nominal.reshape(batch_size, self.m_control))

                dsdt = self.nominal_dynamics(state, u.reshape(batch_size, self.m_control, 1), batch_size)

                dsdt = torch.reshape(dsdt, (batch_size, self.n_state))

                alpha = self.alpha(state)

                dot_h = torch.matmul(grad_h.reshape(batch_size, 1, self.n_state),
                                     dsdt.reshape(batch_size, self.n_state, 1))

                dot_h = dot_h.reshape(batch_size, 1)

                deriv_cond = dot_h + alpha * h

                num_safe = torch.sum(safe_mask)
                num_dang = torch.sum(dang_mask)
                num_mid = torch.sum(mid_mask)

                loss_h_safe = torch.sum(
                    nn.ReLU()(eps - h).reshape(1, batch_size) * safe_mask.reshape(1, batch_size)) / (1e-5 + num_safe)
                loss_h_dang = torch.sum(
                    nn.ReLU()(h + eps).reshape(1, batch_size) * dang_mask.reshape(1, batch_size)) / (1e-5 + num_dang)

                loss_alpha = torch.sum(nn.ReLU()(alpha).reshape(1, batch_size) * safe_mask.reshape(1, batch_size)) / (
                        1e-5 + num_safe)

                acc_h_safe = torch.sum((h >= 0).float() * safe_mask) / (1e-5 + num_safe)
                acc_h_dang = torch.sum((h < 0).float() * dang_mask) / (1e-5 + num_dang)

                loss_deriv_safe = torch.sum(
                    nn.ReLU()(eps_deriv - deriv_cond).reshape(1, batch_size) * safe_mask.reshape(1, batch_size)) / (
                                          1e-5 + num_safe)
                loss_deriv_dang = torch.sum(
                    nn.ReLU()(eps_deriv - deriv_cond).reshape(1, batch_size) * dang_mask.reshape(1, batch_size)) / (
                                          1e-5 + num_dang)
                loss_deriv_mid = torch.sum(
                    nn.ReLU()(eps_deriv - deriv_cond).reshape(1, batch_size) * mid_mask.reshape(1, batch_size)) / (
                                         1e-5 + num_mid)

                acc_deriv_safe = torch.sum((deriv_cond > 0).float() * safe_mask) / (1e-5 + num_safe)
                acc_deriv_dang = torch.sum((deriv_cond > 0).float() * dang_mask) / (1e-5 + num_dang)
                acc_deriv_mid = torch.sum((deriv_cond > 0).float() * mid_mask) / (1e-5 + num_mid)

                loss_action = torch.mean(nn.ReLU()(torch.abs(u - u_nominal) - eps_action))

                loss = loss_h_safe + loss_h_dang + loss_alpha + loss_deriv_safe + loss_deriv_dang + loss_deriv_mid + loss_action * self.action_loss_weight  # + loss_limit1 + loss_limit2 + loss_limit3 +
                # loss_limit4

                self.controller_optimizer.zero_grad()
                self.cbf_optimizer.zero_grad()
                self.alpha_optimizer.zero_grad()

                loss.backward(retain_graph=True)

                self.controller_optimizer.step()
                self.cbf_optimizer.step()
                self.alpha_optimizer.step()

                # log statics
                acc_np[0] += acc_h_safe.detach().cpu().numpy()
                acc_np[1] += acc_h_dang.detach().cpu().numpy()

                acc_np[2] += acc_deriv_safe.detach().cpu().numpy()
                acc_np[3] += acc_deriv_dang.detach().cpu().numpy()
                acc_np[4] += acc_deriv_mid.detach().cpu().numpy()

                loss_np += loss.detach().cpu().numpy()
                loss_h_safe_np += loss_h_safe.detach().cpu().numpy()
                loss_h_dang_np += loss_h_dang.detach().cpu().numpy()
                loss_deriv_safe_np += loss_deriv_safe.detach().cpu().numpy()
                loss_deriv_mid_np += loss_deriv_mid.detach().cpu().numpy()
                loss_deriv_dang_np += loss_deriv_dang.detach().cpu().numpy()
                loss_alpha_np += loss_alpha.detach().cpu().numpy()
                loss_action_np += loss_action.detach().cpu().numpy()
                # loss_limit_np += loss_limit1.detach().cpu().numpy() + loss_limit2.detach().cpu().numpy() +
                # loss_limit3.detach().cpu().numpy() + loss_limit4.detach().cpu().numpy()
            # print(t.toc())

        acc_np /= opt_iter * 10
        loss_np /= opt_iter * 10
        loss_h_safe_np /= opt_iter * 10
        loss_h_dang_np /= opt_iter * 10
        loss_deriv_safe_np /= opt_iter * 10
        loss_deriv_mid_np /= opt_iter * 10
        loss_deriv_dang_np /= opt_iter * 10
        loss_alpha_np /= opt_iter * 10
        loss_action_np /= opt_iter * 10
        # loss_limit_np /= opt_iter * 10

        if self.lr_decay_stepsize >= 0:
            # learning rate decay
            self.cbf_lr_scheduler.step()
            self.controller_lr_scheduler.step()

        return loss_np, acc_np, loss_h_safe_np, loss_h_dang_np, loss_alpha_np, loss_deriv_safe_np, loss_deriv_dang_np, loss_deriv_mid_np, loss_action_np

    def get_mask(self, state):
        """
        args:
            state (bs, n_state)
        returns:
            safe_mask (bs, k_obstacle)
            mid_mask  (bs, k_obstacle)
            dang_mask (bs, k_obstacle)
        """
        safe_mask = self.dyn.safe_mask(state).float()
        dang_mask = self.dyn.unsafe_mask(state).float()
        mid_mask = (1 - safe_mask) * (1 - dang_mask)

        return safe_mask, dang_mask, mid_mask

    def is_safe(self, state):

        # alpha = torch.abs(state[:,1])
        return self.dyn.safe_mask(state)

    def nominal_dynamics(self, state, u, batch_size):
        """
        args:
            state (n_state,)
            u (m_control,)
        returns:
            dsdt (n_state,)
        """

        m_control = self.m_control
        fx = self.dyn._f(state, self.params)
        gx = self.dyn._g(state, self.params)

        for j in range(self.m_control):
            if self.fault == 1 and self.fault_control_index == j:
                u[:, j] = u[:, j].clone().detach().reshape(batch_size, 1)
            else:
                u[:, j] = u[:, j].clone().detach().requires_grad_(True).reshape(batch_size, 1)

        # if self.fault == 1 and self.fault_control_index > -1:
        # u[:,self.fault_control_index] = u[:,self.fault_control_index].detach()

        dsdt = fx + torch.matmul(gx, u)

        return dsdt

    def nominal_controller(self, state, goal, u_norm_max, dyn, constraints):
        """
        args:
            state (n_state,)
            goal (n_state,)
        returns:
            u_nominal (m_control,)
        """
        um, ul = self.dyn.control_limits()

        n_state = self.n_state
        m_control = self.m_control
        params = self.params
        j_const = self.j_const

        size_Q = m_control + j_const

        Q = csc_matrix(10 * identity(size_Q))
        F = np.array([1] * size_Q).reshape(size_Q, 1)

        fx = dyn._f(state, params)
        gx = dyn._g(state, params)

        fx = fx.reshape(n_state, 1)
        gx = gx.reshape(n_state, m_control)

        V, Lg, Lf = constraints.LfLg_new(state, goal, fx, gx, n_state, m_control, j_const, 1, 0.3)

        A = torch.hstack((Lg, V))
        B = Lf

        # for just convergence
        A = torch.tensor((A[1][:]))
        B = torch.tensor(B[1][:])

        # if A[0][-1] == 0:
        #     A = torch.tensor(A[1][:])
        #     B = torch.tensor(B[1][:])

        # if A[-1] == 0 or torch.isnan(torch.sum(A)):
        #     A = []
        #     B = []
        #     u = solve_qp(Q, F, solver="osqp")
        # else:
        # print(A)
        A = scipy.sparse.csc.csc_matrix(A)
        B = np.array(B)
        u = solve_qp(Q, F, A, B, solver="osqp")

        if u is None:
            u = np.array(um) / 2
            u = u.reshape(1, m_control)

        u_nominal = torch.tensor([u[0:self.m_control]]).reshape(1, m_control)

        return u_nominal
