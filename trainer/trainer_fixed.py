import torch
import math
import scipy
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from qpsolvers import solve_qp
from osqp import OSQP
from scipy.sparse import identity
from scipy.sparse import vstack, csr_matrix, csc_matrix
from trainer.FxTS_GF import FxTS_Momentum
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
                 action_loss_weight=0.1,
                 gpu_id=0,
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
        #     self.controller.parameters(), lr=5e-4, momentum=0.2)
        # self.cbf_optimizer = FxTS_Momentum(
        #     self.cbf.parameters(), lr=1e-4, momentum=0.2)
        # self.alpha_optimizer = FxTS_Momentum(
        #     self.alpha.parameters(), lr=5e-4, momentum=0.2)

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
        loss_limit_np = 0.0
        acc_np = np.zeros((5,), dtype=np.float32)
        print("training both CBF and u")
        # t.tic()
        for j in range(10):
            for i in range(opt_iter):
                # t.tic()
                # print(i)
                state, u, u_nominal = self.dataset.sample_data(batch_size, i)
                u_nominal = torch.from_numpy(u_nominal)

                if self.gpu_id >= 0:
                    state = state.cuda(self.gpu_id)
                    u_nominal = u_nominal.cuda(self.gpu_id)
                    self.cbf.to(torch.device('cuda'))
                    self.controller.to(torch.device('cuda'))
                    self.alpha.to(torch.device('cuda'))

                safe_mask, dang_mask, mid_mask = self.get_mask(state)

                u = self.controller(state, u_nominal.reshape(batch_size, self.m_control))
                h, grad_h = self.cbf.V_with_jacobian(state)
                alpha = self.alpha(state)

                dsdt = self.nominal_dynamics(state, u.reshape(batch_size, self.m_control, 1), batch_size)

                dsdt = torch.reshape(dsdt, (batch_size, self.n_state))

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

                loss_alpha = 0.01 * torch.sum(nn.ReLU()(alpha - eps).reshape(1, batch_size) *
                                              safe_mask.reshape(1, batch_size)) / (1e-5 + num_safe)

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
                loss_limit = torch.sum(nn.ReLU()(eps - u[:, 0]))

                loss = loss_h_safe + loss_h_dang + loss_alpha + loss_action * self.action_loss_weight + loss_limit \
                       + loss_deriv_safe + loss_deriv_dang + loss_deriv_mid

                # print("time in loss setup: ")
                # print(t.toc())
                # t.tic()

                self.controller_optimizer.zero_grad()
                self.cbf_optimizer.zero_grad()
                self.alpha_optimizer.zero_grad()

                loss.backward(retain_graph=True)

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
                loss_limit_np += loss_limit.detach().cpu().numpy()
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
        loss_limit_np /= opt_iter * 10

        if self.lr_decay_stepsize >= 0:
            # learning rate decay
            self.cbf_lr_scheduler.step()
            self.controller_lr_scheduler.step()

        return loss_np, acc_np, loss_h_safe_np, loss_h_dang_np, loss_alpha_np, loss_deriv_safe_np, loss_deriv_dang_np, loss_deriv_mid_np, loss_action_np, loss_limit_np

    def train_cbf(self, batch_size=1000, opt_iter=100, eps=0.1, eps_deriv=0.03, k=0.6):
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

        for _ in range(10):
            for i in range(opt_iter):
                # t.tic()
                # print(i)
                state, _, _ = self.dataset.sample_data(batch_size, i)
                device = 'cpu'
                if self.gpu_id >= 0:
                    device = self.gpu_id
                    state = state.cuda(self.gpu_id)
                    self.cbf.to(torch.device(self.gpu_id))
                    self.alpha.to(torch.device(self.gpu_id))

                safe_mask, dang_mask, mid_mask = self.get_mask(state)

                if k < 0.5:
                    h, _ = self.cbf.V_with_jacobian(state)
                    alpha = eps * torch.ones(1, batch_size).to(device) / 10
                    deriv_cond = torch.ones(1, batch_size).to(device)
                else:
                    h, grad_h = self.cbf.V_with_jacobian(state)
                    alpha = self.alpha(state)
                    dot_h_max = self.doth_max(state, grad_h, um, ul)
                    deriv_cond = dot_h_max + alpha.reshape(1, batch_size) * h.reshape(1, batch_size)

                num_safe = torch.sum(safe_mask)
                num_dang = torch.sum(dang_mask)
                num_mid = torch.sum(mid_mask)

                loss_h_safe = torch.sum(
                    nn.ReLU()(eps - h).reshape(1, batch_size) * safe_mask.reshape(1, batch_size)) / (1e-5 + num_safe)
                loss_h_dang = torch.sum(
                    nn.ReLU()(h + eps).reshape(1, batch_size) * dang_mask.reshape(1, batch_size)) / (1e-5 + num_dang)

                loss_alpha = 0.01 * torch.sum(nn.ReLU()(alpha - eps).reshape(1, batch_size) *
                                              safe_mask.reshape(1, batch_size)) / (1e-5 + num_safe)

                acc_h_safe = torch.sum(
                    (h >= 0).reshape(1, batch_size).float() * safe_mask.reshape(1, batch_size)) / (1e-5 + num_safe)
                acc_h_dang = torch.sum(
                    (h < 0).reshape(1, batch_size).float() * dang_mask.reshape(1, batch_size)) / (1e-5 + num_dang)

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

                loss = loss_h_safe + loss_h_dang + loss_alpha + loss_deriv_safe + loss_deriv_dang + loss_deriv_mid

                self.cbf_optimizer.zero_grad()
                self.alpha_optimizer.zero_grad()

                loss.backward(retain_graph=True)

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

        acc_np /= opt_iter * 10
        loss_np /= opt_iter * 10
        loss_h_safe_np /= opt_iter * 10
        loss_h_dang_np /= opt_iter * 10
        loss_deriv_safe_np /= opt_iter * 10
        loss_deriv_mid_np /= opt_iter * 10
        loss_deriv_dang_np /= opt_iter * 10
        loss_alpha_np /= opt_iter * 10

        if self.lr_decay_stepsize >= 0:
            # learning rate decay
            self.cbf_lr_scheduler.step()
            # self.controller_lr_scheduler.step()

        return loss_np, acc_np, loss_h_safe_np, loss_h_dang_np, loss_alpha_np, loss_deriv_safe_np, loss_deriv_dang_np, loss_deriv_mid_np

    def doth_max(self, state, grad_h, um, ul):
        bs = grad_h.shape[0]

        # LhG = LhG.detach().cpu()
        fx = self.dyn._f(state, self.params)
        gx = self.dyn._g(state, self.params)
        if self.gpu_id >= 0:
            fx = fx.cuda(self.gpu_id)
            gx = gx.cuda(self.gpu_id)

        doth = torch.matmul(grad_h, fx)
        LhG = torch.matmul(grad_h, gx)

        sign_grad_h = torch.sign(LhG).reshape(bs, 1, self.m_control)

        if self.fault == 0:
            doth = doth + torch.matmul(sign_grad_h, um.reshape(bs, self.m_control, 1)) + \
                   torch.matmul(1 - sign_grad_h, ul.reshape(bs, self.m_control, 1))
        else:
            for i in range(self.m_control):
                if i == self.fault_control_index:
                    doth = doth - sign_grad_h[:, 0, i].reshape(bs, 1) * um[:, i].reshape(bs, 1) - \
                           (1 - sign_grad_h[:, 0, i].reshape(bs, 1)) * ul[:, i].reshape(bs, 1)
                else:
                    doth = doth + sign_grad_h[:, 0, i].reshape(bs, 1) * um[:, i].reshape(bs, 1) + \
                           (1 - sign_grad_h[:, 0, i].reshape(bs, 1)) * ul[:, i].reshape(bs, 1)

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
        safe_mask = self.dyn.safe_mask(state).float()
        dang_mask = self.dyn.unsafe_mask(state).float()
        mid_mask = (1 - safe_mask) * (1 - dang_mask)

        return safe_mask, dang_mask, mid_mask

    def nominal_dynamics(self, state, u, batch_size):
        """
        args:
            state (n_state,)
            u (m_control,)
        returns:
            dsdt (n_state,)
        """

        fx = self.dyn._f(state, self.params)
        gx = self.dyn._g(state, self.params)

        for j in range(self.m_control):
            if self.fault == 1 and self.fault_control_index == j:
                u[:, j] = u[:, j].clone().detach().reshape(batch_size, 1)
            else:
                u[:, j] = u[:, j].clone().detach().requires_grad_(True).reshape(batch_size, 1)

        dsdt = fx + torch.matmul(gx, u)

        return dsdt
