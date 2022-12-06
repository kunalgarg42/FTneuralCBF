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
                 cbf,
                 dataset,
                 dyn,
                 params,
                 n_state,
                 m_control,
                 goal,
                 rg=0.1,
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
        self.goal = goal
        self.rg = rg
        self.dataset = dataset
        self.fault = fault
        self.fault_control_index = fault_control_index

        if gpu_id >= 0:
            self.goal = self.goal.cuda(gpu_id)
        self.cbf_optimizer = torch.optim.Adam(
            self.cbf.parameters(), lr=1e-4, weight_decay=1e-5)
        # self.cbf_optimizer = FxTS_Momentum(
        #     self.cbf.parameters(), lr=5e-5, momentum=0.2)

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

    def train_cbf(self, batch_size=5000, opt_iter=20, eps=0.1, eps_deriv=0.03):
        loss_np = 0.0
        loss_h_safe_np = 0.0
        loss_h_dang_np = 0.0
        loss_deriv_safe_np = 0.0
        loss_deriv_dang_np = 0.0

        acc_np = np.zeros((4,), dtype=np.float32)
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
        for _ in range(20):
            for i in range(opt_iter):
                # t.tic()
                # print(i)
                state, _, _ = self.dataset.sample_data(batch_size, i)
                if self.gpu_id >= 0:
                    state = state.cuda(self.gpu_id)
                    self.cbf.to(torch.device(self.gpu_id))

                safe_mask, dang_mask = self.get_mask(state)

                h, grad_h = self.cbf.V_with_jacobian(state)

                dot_h_max = self.doth_max(h, state, grad_h, um, ul)
                deriv_cond = dot_h_max  # + alpha.reshape(1, batch_size) * h.reshape(1, batch_size)

                num_safe = torch.sum(safe_mask)
                num_dang = torch.sum(dang_mask)

                acc_h_safe = torch.sum(
                    (h < 0).reshape(1, batch_size).float() * safe_mask.reshape(1, batch_size)) / (1e-5 + num_safe)
                acc_h_dang = torch.sum(
                    (h >= 0).reshape(1, batch_size).float() * dang_mask.reshape(1, batch_size)) / (1e-5 + num_dang)

                loss_h_safe = 10 * torch.sum(
                    nn.ReLU()(h + eps).reshape(1, batch_size) * safe_mask.reshape(1, batch_size)) / (
                                      1e-5 + num_safe) / (acc_h_safe.clone().detach() + 1e-5)

                loss_h_dang = 10 * torch.sum(
                    nn.ReLU()(eps - h).reshape(1, batch_size) * dang_mask.reshape(1, batch_size)) / (
                                      1e-5 + num_dang) / (acc_h_dang.clone().detach() + 1e-5)

                acc_deriv_safe = torch.sum((deriv_cond < 0).reshape(1, batch_size) .float() * safe_mask.reshape(1, batch_size) ) / (1e-5 + num_safe)
                acc_deriv_dang = torch.sum((deriv_cond < 0).reshape(1, batch_size) .float() * dang_mask.reshape(1, batch_size) ) / (1e-5 + num_dang)

                loss_deriv_safe = torch.sum(
                    nn.ReLU()(eps_deriv + deriv_cond).reshape(1, batch_size) * safe_mask.reshape(1, batch_size)) / (
                                          1e-5 + num_safe) / (acc_deriv_safe.detach() + 1e-5)
                loss_deriv_dang = torch.sum(
                    nn.ReLU()(eps_deriv + deriv_cond).reshape(1, batch_size) * dang_mask.reshape(1, batch_size)) / (
                                          1e-5 + num_dang) / (acc_deriv_dang.detach() + 1e-5)

                loss = loss_h_safe + loss_h_dang + loss_deriv_safe + loss_deriv_dang

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

                loss_np += loss.detach().cpu().numpy()
                loss_h_safe_np += loss_h_safe.detach().cpu().numpy()
                loss_h_dang_np += loss_h_dang.detach().cpu().numpy()
                loss_deriv_safe_np += loss_deriv_safe.detach().cpu().numpy()
                loss_deriv_dang_np += loss_deriv_dang.detach().cpu().numpy()

        acc_np /= opt_iter * 20
        loss_np /= opt_iter * 20
        loss_h_safe_np /= opt_iter * 20
        loss_h_dang_np /= opt_iter * 20
        loss_deriv_safe_np /= opt_iter * 20
        loss_deriv_dang_np /= opt_iter * 20

        if self.lr_decay_stepsize >= 0:
            # learning rate decay
            self.cbf_lr_scheduler.step()
            # self.controller_lr_scheduler.step()

        return loss_np, acc_np, loss_h_safe_np, loss_h_dang_np, loss_deriv_safe_np, loss_deriv_dang_np

    def doth_max(self, h, state, grad_h, um, ul):
        bs = grad_h.shape[0]

        # LhG = LhG.detach().cpu()
        fx = self.dyn._f(state, self.params)
        gx = self.dyn._g(state, self.params)
        vec_ones = 10 * torch.ones(bs, 2)
        if self.gpu_id >= 0:
            fx = fx.cuda(self.gpu_id)
            gx = gx.cuda(self.gpu_id)
            vec_ones = vec_ones.cuda(self.gpu_id)

        doth = torch.matmul(grad_h, fx)
        h1 = torch.sign(h) * torch.abs(h) ** 1.2
        h2 = torch.sign(h) * torch.abs(h) ** 0.8

        LhG = torch.matmul(grad_h, gx).reshape(bs, self.m_control)
        LhG = torch.hstack((LhG, h1, h2))
        # LhG = torch.hstack((LhG, h2))

        sign_grad_h = torch.sign(LhG).reshape(bs, 1, self.m_control + 2)

        um = torch.hstack((um, vec_ones)).reshape(self.m_control + 2, bs)
        ul = torch.hstack((ul, vec_ones)).reshape(self.m_control + 2, bs)

        uin = - um.reshape(self.m_control + 2, bs) * \
              (sign_grad_h > 0).reshape(self.m_control + 2, bs) + ul.reshape(self.m_control + 2, bs) * (
                      sign_grad_h <= 0).reshape(self.m_control + 2, bs)

        doth = doth + torch.matmul(torch.abs(LhG).reshape(bs, 1, self.m_control + 2),
                                   uin.reshape(bs, self.m_control + 2, 1))

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
        bs = state.shape[0]
        state_z = state[:, 2].reshape(bs, 1)
        device_id = state_z.get_device()

        dist_g = torch.linalg.norm(state_z-self.goal[0, 2], dim=1).reshape(bs, 1)

        if device_id >= 0:
            safe_mask = torch.ones(bs, 1).cuda(self.gpu_id).logical_and(dist_g < self.rg)
            dang_mask = torch.ones(bs, 1).cuda(self.gpu_id).logical_and(dist_g >= self.rg)
        else:
            safe_mask = torch.ones(bs, 1).logical_and(dist_g < self.rg)
            dang_mask = torch.ones(bs, 1).logical_and(dist_g >= self.rg)

        return safe_mask, dang_mask

    def get_mask_cpu(self, state, goal, rg):
        """
        args:
            state (bs, n_state)
        returns:
            safe_mask (bs, k_obstacle)
            mid_mask  (bs, k_obstacle)
            dang_mask (bs, k_obstacle)
        """
        bs = state.shape[0]
        state_z = state[:, 2].reshape(bs, 1)

        dist_g = torch.linalg.norm(state_z-goal[0, 2], dim=1).reshape(bs, 1)

        safe_mask = torch.ones(bs, 1).logical_and(dist_g < rg)
        dang_mask = torch.ones(bs, 1).logical_and(dist_g >= rg)

        return safe_mask, dang_mask


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
