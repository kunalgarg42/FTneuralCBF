import torch
import math
import scipy
import numpy as np
from qpsolvers import solve_qp
from osqp import OSQP
from scipy.sparse import identity
from scipy.sparse import vstack, csr_matrix, csc_matrix
import torch.distributions as td
from trainer.constraints_fw import LfLg_new

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

    def nominal_controller(self, state, goal, dyn):
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

        V, Lg, Lf = LfLg_new(state, goal, fx, gx, n_state, m_control)

        A = torch.hstack((Lg, V))
        B = Lf

        A = scipy.sparse.csc.csc_matrix(A)
        B = np.array(B)

        u = solve_qp(Q, F, A, B, solver="osqp")

        if u is None:
            u = np.array(um.clone()) / 2
            u = u.reshape(1, m_control)

        u_nominal = torch.tensor([u[0:self.m_control]]).reshape(1, m_control)

        return u_nominal

    def neural_controller(self, u_nominal, fx, gx, h, grad_h):
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
        F = torch.hstack((torch.tensor(u_nominal).reshape(m_control), torch.tensor(1.0))).reshape(size_Q, 1)

        F = - np.array(F)

        Lg = torch.matmul(grad_h, gx)
        Lf = torch.matmul(grad_h, fx)

        A = torch.hstack((- Lg.reshape(1, m_control), - h.reshape(1, 1)))
        A = torch.tensor(A.detach().cpu())
        A = torch.vstack((A, - 1 * torch.eye(size_Q)))

        # print(A.shape)

        B = Lf.detach().cpu().numpy()
        B = np.array(B).reshape(j_const)
        B = np.vstack((B, np.array([0] * size_Q).reshape(size_Q, 1)))
        B[-1] = -1000000.0

        A = scipy.sparse.csc.csc_matrix(A)
        u = solve_qp(Q, F, G=A, h=B, solver="osqp")

        if u is None:
            u = np.array(um.clone()) / 2
            u = u.reshape(1, m_control)

        u_neural = torch.tensor([u[0:self.m_control]]).reshape(1, m_control)

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

        samples.scatter_(1, normal_idx[:, None], tmp)

        return samples

    def x_samples(self, sm, sl, batch):
        """
        args:
            state lower limit sl
            state upper limit sm
        returns:
            samples on boundary x
        """

        n_dims = self.n_state

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

        return samples

    def doth_max(self, grad_h, gx, um, ul):
        bs = grad_h.shape[0]

        LhG = torch.matmul(grad_h, gx)

        sign_grad_h = torch.sign(LhG).reshape(bs, 1, self.m_control)
        if self.fault == 0:
            doth = torch.matmul(sign_grad_h, um.reshape(bs, self.m_control, 1)) + \
                   torch.matmul(1 - sign_grad_h, ul.reshape(bs, self.m_control, 1))
        else:
            doth = torch.zeros(bs, 1)
            for i in range(self.m_control):
                if i == self.fault_control_index:
                    doth = doth - sign_grad_h[:, 0, i].reshape(bs, 1) * um[:, i].reshape(bs, 1) - \
                           (1 - sign_grad_h[:, 0, i].reshape(bs, 1)) * ul[:, i].reshape(bs, 1)
                else:
                    doth = doth + sign_grad_h[:, 0, i].reshape(bs, 1) * um[:, i].reshape(bs, 1) + \
                           (1 - sign_grad_h[:, 0, i].reshape(bs, 1)) * ul[:, i].reshape(bs, 1)

        return doth.reshape(1, bs)
