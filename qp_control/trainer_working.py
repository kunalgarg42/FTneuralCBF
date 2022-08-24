import torch
import math
import scipy
from torch import nn
import numpy as np
from qpsolvers import solve_qp
from osqp import OSQP
from scipy.sparse import identity
from scipy.sparse import vstack, csr_matrix, csc_matrix

from qp_control.FxTS_GF import FxTS_Momentum 

class Trainer(object):

    def __init__(self, 
                 # state, 
                 # obstacle, 
                 # goal,
                 controller, 
                 cbf, 
                 alpha,
                 dataset,
                 dyn, 
                 n_pos,
                 params, 
                 n_state,
                 m_control,
                 j_const = 1,
                 dt=0.05, 
                 safe_alpha=0.3, 
                 dang_alpha=0.4, 
                 action_loss_weight=0.08,
                 gpu_id=-1,
                 lr_decay_stepsize=-1):
        

        # self.state = state
        # self.obstacle = obstacle
        # self.goal = goal
        self.params = params
        self.n_state = n_state
        self.m_control = m_control
        self.j_const = j_const
        self.controller = controller
        self.dyn = dyn
        self.cbf = cbf
        self.alpha = alpha
        self.dataset = dataset
        # self.nominal_dynamics = nominal_dynamics
        
        self.controller_optimizer = torch.optim.Adam(
            self.controller.parameters(), lr=5e-4, weight_decay=1e-5)
        self.cbf_optimizer = torch.optim.Adam(
            self.cbf.parameters(), lr=1e-4, weight_decay=1e-5)
        self.alpha_optimizer = torch.optim.Adam(
            self.alpha.parameters(),lr = 1e-4, weight_decay = 1e-5)
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


    def train_cbf(self, batch_size=256, opt_iter=500, eps=0.1):

        loss_np = 0.0
        acc_np = np.zeros((5,), dtype=np.float32)
        
        for i in range(opt_iter):
            # state (bs, n_state), obstacle (bs, k_obstacle, n_state)
            # u_nominal (bs, m_control), state_next (bs, n_state)
            state, obstacle, u_nominal, state_next, state_error = self.dataset.sample_data(batch_size)
            state = torch.from_numpy(state)
            obstacle = torch.from_numpy(obstacle)
            state_next = torch.from_numpy(state_next)

            if self.gpu_id >= 0:
                state = state.cuda(self.gpu_id)
                obstacle = obstacle.cuda(self.gpu_id)
                state_next = state_next.cuda(self.gpu_id)

            safe_mask, dang_mask, mid_mask = self.get_mask(state, obstacle)
            h = self.cbf(state, obstacle)
            alpha  = self.alpha(state,obstacle)

            num_safe = torch.sum(safe_mask)
            num_dang = torch.sum(dang_mask)
            num_mid = torch.sum(mid_mask)

            loss_h_safe = torch.sum(nn.ReLU()(eps - h) * safe_mask) / (1e-5 + num_safe)
            loss_h_dang = torch.sum(nn.ReLU()(h + eps) * dang_mask) / (1e-5 + num_dang)

            acc_h_safe = torch.sum((h >= 0).float() * safe_mask) / (1e-5 + num_safe)
            acc_h_dang = torch.sum((h < 0).float() * dang_mask) / (1e-5 + num_dang)

            h_next = self.cbf(state_next, obstacle)
            deriv_cond = (h_next - h) / self.dt + alpha*h

            loss_deriv_safe = torch.sum(nn.ReLU()(-deriv_cond) * safe_mask) / (1e-5 + num_safe)
            loss_deriv_dang = torch.sum(nn.ReLU()(-deriv_cond) * dang_mask) / (1e-5 + num_dang)
            loss_deriv_mid = torch.sum(nn.ReLU()(-deriv_cond) * mid_mask) / (1e-5 + num_mid)

            acc_deriv_safe = torch.sum((deriv_cond > 0).float() * safe_mask) / (1e-5 + num_safe)
            acc_deriv_dang = torch.sum((deriv_cond > 0).float() * dang_mask) / (1e-5 + num_dang)
            acc_deriv_mid = torch.sum((deriv_cond > 0).float() * mid_mask) / (1e-5 + num_mid)

            loss = loss_h_safe + loss_h_dang + loss_deriv_safe + loss_deriv_dang + loss_deriv_mid

            self.cbf_optimizer.zero_grad()
            loss.backward()
            self.cbf_optimizer.step()

            # log statics
            acc_np[0] += acc_h_safe.detach().cpu().numpy()
            acc_np[1] += acc_h_dang.detach().cpu().numpy()

            acc_np[2] += acc_deriv_safe.detach().cpu().numpy()
            acc_np[3] += acc_deriv_dang.detach().cpu().numpy()
            acc_np[4] += acc_deriv_mid.detach().cpu().numpy()

            loss_np += loss.detach().cpu().numpy()

        acc_np = acc_np / opt_iter
        loss_np = loss_np / opt_iter
        return loss_np, acc_np

        
    def train_controller(self, batch_size=256, opt_iter=50, eps=0.1):

        loss_np = 0.0
        acc_np = np.zeros((3,), dtype=np.float32)

        for i in range(opt_iter):
            # state (bs, n_state), obstacle (bs, k_obstacle, n_state)
            # u_nominal (bs, m_control), state_next (bs, n_state)
            state, obstacle, u_nominal, state_next, state_error = self.dataset.sample_data(batch_size)
            state = torch.from_numpy(state)
            obstacle = torch.from_numpy(obstacle)
            u_nominal = torch.from_numpy(u_nominal)
            state_next = torch.from_numpy(state_next)
            state_error = torch.from_numpy(state_error)


            if self.gpu_id >= 0:
                state = state.cuda(self.gpu_id)
                obstacle = obstacle.cuda(self.gpu_id)
                u_nominal = u_nominal.cuda(self.gpu_id)
                state_next = state_next.cuda(self.gpu_id)
                state_error = state_error.cuda(self.gpu_id)

            safe_mask, dang_mask, mid_mask = self.get_mask(state, obstacle)

            h = self.cbf(state, obstacle)


            alpha  = self.alpha(state,obstacle)

            u = self.controller(state, obstacle, u_nominal, state_error)

            dsdt_nominal = self.nominal_dynamics(state, u)
            
            # print(dsdt.shape)

            state_next_nominal = state + dsdt_nominal * self.dt

            state_next_with_grad = state_next_nominal + (state_next - state_next_nominal).detach()

            h_next = self.cbf(state_next_with_grad, obstacle)
            deriv_cond = (h_next - h) / self.dt + alpha*h

            num_safe = torch.sum(safe_mask)
            num_dang = torch.sum(dang_mask)
            num_mid = torch.sum(mid_mask)

            loss_deriv_safe = torch.sum(nn.ReLU()(-deriv_cond) * safe_mask) / (1e-5 + num_safe)
            loss_deriv_dang = torch.sum(nn.ReLU()(-deriv_cond) * dang_mask) / (1e-5 + num_dang)
            loss_deriv_mid = torch.sum(nn.ReLU()(-deriv_cond) * mid_mask) / (1e-5 + num_mid)

            acc_deriv_safe = torch.sum((deriv_cond > 0).float() * safe_mask) / (1e-5 + num_safe)
            acc_deriv_dang = torch.sum((deriv_cond > 0).float() * dang_mask) / (1e-5 + num_dang)
            acc_deriv_mid = torch.sum((deriv_cond > 0).float() * mid_mask) / (1e-5 + num_mid)

            loss_action = torch.mean((u - u_nominal)**2)

            loss = loss_deriv_safe + loss_deriv_dang + loss_deriv_mid + loss_action * self.action_loss_weight

            self.controller_optimizer.zero_grad()
            loss.backward()
            self.controller_optimizer.step()

            # log statics
            acc_np[0] += acc_deriv_safe.detach().cpu().numpy()
            acc_np[1] += acc_deriv_dang.detach().cpu().numpy()
            acc_np[2] += acc_deriv_mid.detach().cpu().numpy()

            loss_np += loss.detach().cpu().numpy()

        acc_np = acc_np / opt_iter
        loss_np = loss_np / opt_iter
        return loss_np, acc_np


    def nominal_dynamics(self, state, u,batch_size):
        """
        args:
            state (n_state,)
            u (m_control,)
        returns:
            dsdt (n_state,)
        """
        m_control = self.m_control
        fx = self.dyn._f(state,self.params)
        gx = self.dyn._g(state, self.params)
        u = torch.tensor(u).reshape(batch_size,m_control,1)

        # dsdt = np.array(dsdt,dtype = np.float32)
        # dsdt = torch.from_numpy(dsdt)
        # print(dsdt.shape)


        # fx = torch.from_numpy(fx)
        # gx = torch.from_numpy(gx)
        # print(fx.shape)
        # print(gx.shape)
        # print(u.shape)


        dsdt = fx + torch.matmul(gx,u)
        # print(dsdt.shape)

        # print(asas)

        return dsdt

    def train_cbf_and_controller(self, batch_size=1024, opt_iter=100, eps=0.1, eps_deriv=0.03, eps_action=0.2):

        loss_np = 0.0
        acc_np = np.zeros((5,), dtype=np.float32)

        for i in range(opt_iter):
            # state (bs, n_state), obstacle (bs, k_obstacle, n_state)
            # u_nominal (bs, m_control), state_next (bs, n_state)
            grad_h, state, u_nominal, state_next, state_error = self.dataset.sample_data(batch_size)
            grad_h = torch.from_numpy(grad_h)
            # state = torch.from_numpy(state)
            u_nominal = torch.from_numpy(u_nominal)
            # state_next = torch.from_numpy(state_next)
            # state_error = torch.from_numpy(state_error)
            # batch_size = np.array(state).shape[0]
            # print(np.array(state).shape)

            if self.gpu_id >= 0:
                state = state.cuda(self.gpu_id)
                u_nominal = u_nominal.cuda(self.gpu_id)
                state_next = state_next.cuda(self.gpu_id)
                state_error = state_error.cuda(self.gpu_id)

            safe_mask, dang_mask, mid_mask = self.get_mask(state)


            h = self.cbf(state)

            # print(h.shape)



            alpha  = self.alpha(state)
            # print(u_nominal.shape)
            # alpha = 1

            u = self.controller(state,u_nominal)


            # print(u.shape)

            dsdt_nominal = self.nominal_dynamics(state, u, batch_size)
            # print(dsdt_nominal.shape)
            
            dsdt_nominal = torch.reshape(dsdt_nominal,(batch_size,self.n_state))
            # print(dsdt_nominal.shape)
            

            state_next_nominal = state + dsdt_nominal * self.dt

            


            state_next_with_grad = state_next #state_next_nominal + (state_next - state_next_nominal).detach()

            # print(state.shape)
            # print(state_next)
            # print(state_next_nominal)
            # print(state_next_with_grad)

            # state_next_with_grad = torch.from_numpy(state_next_with_grad)
 
            h_next = self.cbf(state_next_with_grad)

            # print(h_next[1])
            # state0 = state.clone().detach().requires_grad_(True)

            # grad_h = torch.autograd.grad(h,state0, allow_unused=True)

            # print(grad_h.shape)

            dot_h = torch.matmul(grad_h.reshape(batch_size,1, self.n_state),dsdt_nominal.reshape(batch_size,self.n_state,1))
            dot_h = dot_h.reshape(batch_size,1)
            # print(dot_h.shape)

            # print(asasasa)
            # h_next = self.cbf(state,obstacle)
            # deriv_cond = (h_next - h) / self.dt + alpha*h
            deriv_cond = dot_h + alpha * h

            # state0 = state
            # state0 = state0.clone().detach().requires_grad_(True)
            # h1 = self.cbf(state0,obstacle)

            # grad_h = torch.autograd.grad(h1,state0)
            # grad_h = grad_h[0]
            # grad_h = np.array(grad_h,dtype= float).reshape(1,4)
            # dsdt0 = dsdt_nominal
            # dsdt0 = dsdt0.detach().numpy()
            # dsdt0 = np.array(dsdt0,dtype = float).reshape(4,1)
            # nab_h = np.dot(grad_h, dsdt0)

            # # print(nab_h)

            # # print((h_next-h).detach().numpy() / self.dt)

            # # print(grad_h[0][0])

            # state1 = (state_next_with_grad + state).clone().detach().requires_grad_(True) / 2.0
            # h2 = self.cbf(state1,obstacle)

            # grad_h = torch.autograd.grad(h2,state1)
            # grad_h = grad_h[0]
            # grad_h = np.array(grad_h,dtype= float).reshape(1,4)
            # # dsdt0 = state0.detach().numpy()
            # # dsdt0 = np.array(dsdt0,dtype = float).reshape(4,1)
            # # nab_h = np.dot(grad_h,state0)
            # print(grad_h[0][0])

            # x_err = state_next_with_grad - state
            # x_err = x_err.detach().numpy()
            # x_err = np.array(x_err).reshape(4,1)

            # # print(x_err)

            # grad_h_0 = (h_next-h).detach().numpy()/x_err[0][0]
            # grad_h_0 = np.array(grad_h_0,dtype = float)
            # print(grad_h_0)

            # print((h_next-h)/x_err[1][0])

            # print((h_next-h)/x_err[2][0])

            # print((h_next-h)/x_err[3][0])

            # print(hasdass)

            num_safe = torch.sum(safe_mask)
            num_dang = torch.sum(dang_mask)
            num_mid = torch.sum(mid_mask)

            # print(num_safe)
            # print(num_dang)
            # print(num_mid)

            loss_h_safe = torch.sum(nn.ReLU()(eps - h) * safe_mask) / (1e-5 + num_safe)
            loss_h_dang = torch.sum(nn.ReLU()(h + eps) * dang_mask) / (1e-5 + num_dang)

            loss_alpha = torch.sum(nn.ReLU()(alpha) * safe_mask) / (1e-5 + num_safe)

            acc_h_safe = torch.sum((h >= 0).float() * safe_mask) / (1e-5 + num_safe)
            acc_h_dang = torch.sum((h < 0).float() * dang_mask) / (1e-5 + num_dang)

            loss_deriv_safe = torch.sum(nn.ReLU()(eps_deriv - deriv_cond) * safe_mask) / (1e-5 + num_safe)
            loss_deriv_dang = torch.sum(nn.ReLU()(eps_deriv - deriv_cond) * dang_mask) / (1e-5 + num_dang)
            loss_deriv_mid = torch.sum(nn.ReLU()(eps_deriv - deriv_cond) * mid_mask) / (1e-5 + num_mid)

            acc_deriv_safe = torch.sum((deriv_cond > 0).float() * safe_mask) / (1e-5 + num_safe)
            acc_deriv_dang = torch.sum((deriv_cond > 0).float() * dang_mask) / (1e-5 + num_dang)
            acc_deriv_mid = torch.sum((deriv_cond > 0).float() * mid_mask) / (1e-5 + num_mid)

            loss_action = torch.mean(nn.ReLU()(torch.abs(u - u_nominal) - eps_action))



            loss = loss_h_safe + loss_h_dang + loss_alpha + loss_deriv_safe + loss_deriv_dang + loss_deriv_mid + loss_action * self.action_loss_weight

            # loss.set_anomaly_detect(True)
            self.controller_optimizer.zero_grad()
            self.cbf_optimizer.zero_grad()
            self.alpha_optimizer.zero_grad()

            # loss.retain_grad()

            loss.backward(retain_graph = True)

            loss_temp = loss

            if math.isnan(loss_temp.detach().cpu().numpy()):
                continue

            # loss = loss_temp

            # for j in range(100):

            # grad_h = self.cbf_optimizer.parameters().grad
            

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

        acc_np = acc_np / opt_iter
        loss_np = loss_np / opt_iter

        if self.lr_decay_stepsize >= 0:
            # learning rate decay
            self.cbf_lr_scheduler.step()
            self.controller_lr_scheduler.step()
        
        return loss_np, acc_np, loss_h_safe, loss_h_dang, loss_alpha, loss_deriv_safe , loss_deriv_dang , loss_deriv_mid , loss_action


    def get_mask(self, state):
        """
        args:
            state (bs, n_state)
            obstacle (bs, k_obstacle, n_state)
        returns:
            safe_mask (bs, k_obstacle)
            mid_mask  (bs, k_obstacle)
            dang_mask (bs, k_obstacle)
        """
        # state = torch.unsqueeze(state, 2)[:, :self.n_pos]    # (bs, n_pos, 1)
        alpha_1 = state[:,1]
        # alpha_1 = alpha.detach().cpu().numpy()
        alpha_1 = torch.abs(alpha_1)
        # print(alpha_1)

        # print(saas)
        # obstacle = obstacle.permute(0, 2, 1)[:, :self.n_pos] # (bs, n_pos, k_obstacle)
        # dist = torch.norm(state, dim=1)

        safe_mask = (alpha_1 <= self.safe_alpha).float()
        dang_mask = (alpha_1 >= self.dang_alpha).float()
        mid_mask = (1 - safe_mask) * (1 - dang_mask)

        return safe_mask, dang_mask, mid_mask

    def is_safe(self, state):

        alpha = torch.abs(state[:,1])
        return alpha <= self.safe_alpha

        
    def nominal_controller(self, state, goal, u_norm_max, dyn,constraints):
        """
        args:
            state (n_state,)
            goal (n_state,)
        returns:
            u_nominal (m_control,)
        """
        n_state = self.n_state
        m_control = self.m_control
        params = self.params
        j_const = self.j_const

        size_Q = m_control + j_const

        Q = csc_matrix(10*identity(size_Q))
        F = np.array([1]*size_Q).reshape(size_Q,1)
        # state = np.array(state).reshape(n_state,1)
        # goal = np.array(goal).reshape(n_state,1)
        # V = np.linalg.norm(state-goal)
        # V = V**2-0.1

        fx = dyn._f(state,params)
        gx = dyn._g(state,params)

        fx = fx.reshape(n_state,1)
        gx = gx.reshape(n_state,m_control)

        V, Lg, Lf = constraints.LfLg_new(state,goal,fx,gx,n_state, m_control, j_const, 1)
        # A = np.hstack((Lg, V))
        # A = np.array(A,dtype = float)
        # Lg = torch.tensor(Lg)
        # V = torch.tensor(V)
        # print(Lg.shape)
        # print(V.shape)
        A = torch.hstack((Lg, V))
        B = Lf

        # A_init, B_init = A, B
        # N = np.array(A_init).shape[0]
        # m = np.array(gx).shape[1]
        # A = []
        # B = []

        # print(A)

        # print(B)

        # print(asasdadasda)
        if A[0][-1] == 0:
            A = torch.tensor(A[1][:])
            B = torch.tensor(B[1][:])
            # print(A)
            # print(B)
            

        # print(A[-1])

        if A[-1] == 0 or torch.isnan(torch.sum(A)):
            A = []
            B = []
            u = solve_qp(Q, F, solver = "osqp")
        else:
            # print(A)
            A = scipy.sparse.csc.csc_matrix(A)
            u = solve_qp(Q, F, A, B, solver="osqp")

            # print(B)
        # print(asasa)
        # act_cont = []
        # for i in range(N):
        #     if abs(A_init[i][0])+abs(A_init[i][1])>0.0001:
        #         A.append(A_init[i])
        #         B.append(B_init[i])
        
        # A = np.array(A)
        
        # B = np.array(B)
        # print(B)
        # 
        # if B.shape[0]<1:
            # u = solve_qp(Q, F, solver = "osqp")
        # else:
        

        # print(u)

        u_nominal = torch.tensor([u[0], u[1], u[2], u[3]]).reshape(1,m_control)
        # print(u_nominal)
        # K = np.array([[1, 1, 0, 0],
                    # [0, -1, 1, 1]])
        # u_nominal = -K.dot(state - goal)
        # u_nominal = np.array(u_nominal, dtype = float).reshape(1,2)
        # print(u_nominal)

        norm = torch.linalg.norm(u_nominal)

        if norm > u_norm_max:
            u_nominal = u_nominal / norm * u_norm_max
        return u_nominal

    def get_noise():
        noise = np.array([0]*4).reshape(4,1)
        if np.random.uniform() < 0.05:
            noise = np.random.normal(size=(4,)) * 0.5
        noise = np.copy(noise)
        noise[:3] = 0
        return noise