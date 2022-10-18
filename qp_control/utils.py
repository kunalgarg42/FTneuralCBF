import pdb

import torch
import torch.distributions as td
import math
import scipy
import numpy as np
from qpsolvers import solve_qp
from osqp import OSQP
from scipy.sparse import identity
from scipy.sparse import vstack, csr_matrix, csc_matrix


class Utils(object):

    def __init__(self,
                 dyn,
                 params,
                 n_state,
                 m_control,
                 j_const=1,
                 dt=0.05,
                 fault=0,
                 fault_control_index=-1):

        self.params = params
        self.n_state = n_state
        self.m_control = m_control
        self.j_const = j_const
        self.dyn = dyn
        self.fault = fault
        self.fault_control_index = fault_control_index
        self.dt = dt

    def is_safe(self, state):

        # alpha = torch.abs(state[:,1])
        return self.dyn.safe_mask(state)

    def is_unsafe(self, state):

        # alpha = torch.abs(state[:,1])
        return self.dyn.unsafe_mask(state)

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

        dsdt = fx + torch.matmul(gx, u)

        return dsdt

    def nominal_controller(self, state, goal, u_n, dyn, constraints):
        """
        args:
            state (n_state,)
            goal (n_state,)
        returns:
            u_nominal (m_control,)
        """
        um, ul = self.dyn.control_limits()
        sm, sl = self.dyn.state_limits()

        n_state = self.n_state
        m_control = self.m_control
        params = self.params
        j_const = self.j_const

        batch_size = state.shape[0]

        size_Q = m_control + j_const

        Q = csc_matrix(identity(size_Q))
        Q[0, 0] = 1 / um[0]

        F = torch.ones(size_Q, 1)
        u_nominal = u_n
        for i in range(batch_size):
            state_i = state[i, :].reshape(1,n_state)
            F[0:m_control] = - u_n[i, :].reshape(m_control, 1)
            F = np.array(F)
            F[0] = F[0] / um[0]
            F[-1] = - 10
            fx = dyn._f(state_i, params)
            gx = dyn._g(state_i, params)

            fx = fx.reshape(n_state, 1)
            gx = gx.reshape(n_state, m_control)

            V, Lg, Lf = constraints.LfLg_new(state_i, goal, fx, gx, n_state, m_control, j_const, 1, [np.pi / 8, -np.pi / 80])

            # if V == 0:
            #     V[] = 1e-4

            A = torch.hstack((- Lg, - V))
            B = Lf

            G = scipy.sparse.csc.csc_matrix(A)
            h = - np.array(B)
            # u = scipy.optimize.linprog(F, A_ub=G, b_ub=h)
            u = solve_qp(Q, F, G, h, solver="osqp")

            if u is None:
                u = u_n[i, :].reshape(1, m_control)
            #     u = u.reshape(1,m_control)
            u = u[0:m_control]
            # print(u)
            u_nominal[i, :] = torch.tensor(u).reshape(1, m_control)

        return u_nominal

    def neural_controller(self, u_nominal, fx, gx, h, grad_h, fault_start):
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

        size_Q = m_control + 1

        Q = csc_matrix(identity(size_Q))
        # Q[0,0] = 1 / um[0]
        F = torch.hstack((torch.tensor(u_nominal).reshape(m_control), torch.tensor(1.0))).reshape(size_Q, 1)

        F = - np.array(F)
        # F[0] = F[0] / um[0]

        Q = Q / 100
        F = F / 100

        F[-1] = -1

        Lg = torch.matmul(grad_h, gx)
        Lf = torch.matmul(grad_h, fx)

        if fault_start == 1:
            # Lg[]
            # print(Lg.shape)
            Lf = Lf - torch.abs(Lg[0, 0, self.fault_control_index]) * um[self.fault_control_index]
            Lg[0, 0, self.fault_control_index] = 0

        if h == 0:
            h = 1e-4

        A = torch.hstack((- Lg.reshape(1, m_control), -h))
        A = torch.tensor(A.detach().cpu())
        B = Lf.detach().cpu().numpy()
        B = np.array(B)

        # print(A)
        A = scipy.sparse.csc.csc_matrix(A)
        u = solve_qp(Q, F, A, B, solver="osqp")

        # print(A)
        # print(B)
        # print(u)
        # print(asasa)

        if u is None:
            u_neural = u_nominal.reshape(m_control)
        else:
            u_neural = torch.tensor([u[0:self.m_control]]).reshape(1, m_control)
            # u = np.array(um.clone()) / 2
        #     u = u.reshape(1,m_control)
        # print(u.shape)

        return u_neural

    def x_bndr(self, sm, sl, N):
        """
        args:
            state lower limit sl
            state upper limit sm
        returns:
            samples on boundary x
        """

        n_dims = self.n_state
        batch = N

        normal_idx = torch.randint(0, n_dims, size=(batch,))
        assert normal_idx.shape == (batch,)

        # 2: Choose whether it takes the value of hi or lo.
        direction = torch.randint(2, size=(batch,), dtype=torch.bool)
        assert direction.shape == (batch,)

        lo = sl
        hi = sm
        assert lo.shape == hi.shape == (n_dims,)
        dist = td.Uniform(lo, hi)

        samples = dist.sample((batch,))
        assert samples.shape == (batch, n_dims)

        tmp = torch.where(direction, hi[normal_idx], lo[normal_idx])
        assert tmp.shape == (batch,)

        # print(tmp.shape)
        # tmp = 13 * torch.ones(batch)
        tmp = tmp[:, None].repeat(1, n_dims)

        # print("samples")
        # print(samples)
        # print("samples2")
        # print(samples[:, normal_idx])
        # print(samples[:, normal_idx].shape)

        # tmp2 = torch.arange(batch * n_dims).reshape((batch, n_dims)).float()

        # print(normal_idx)
        # print(samples.shape)

        # samples[:, normal_idx] = tmp
        samples.scatter_(1, normal_idx[:, None], tmp)

        # eq_lo = samples == sl
        # eq_hi = samples == sm
        #
        # n_on_bdry = torch.sum(eq_lo, dim=1) + torch.sum(eq_hi, dim=1)
        # all_on_bdry = torch.all(n_on_bdry >= 1)
        # print("all_on_bdry: ", all_on_bdry)

        return samples

