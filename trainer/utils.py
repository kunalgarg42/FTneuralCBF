import pdb
import torch
import torch.distributions as td
import math
import scipy
import numpy as np
import osqp
from qpsolvers import solve_qp
from osqp import OSQP
from scipy.sparse import identity
from scipy.sparse import vstack, csr_matrix, csc_matrix
from pytictoc import TicToc
from trainer.constraints_crazy import LfLg_new

# from qpth.qp import QPFunction

t = TicToc()

m = osqp.OSQP()

P = torch.eye(1250)
q = np.array([1]*1250).reshape(1250)
A = torch.ones(11, 1250)
u = np.array([1]*11).reshape(11)

P = csc_matrix(P)
A = scipy.sparse.csc.csc_matrix(A)

m.setup(P=P, q=q, A=A, u=u, verbose=False)

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
        return self.dyn.safe_mask(state, self.fault)

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

    def nominal_controller(self, state, goal, u_n, dyn):
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
            # t.tic()
            state_i = state[i, :].reshape(1, n_state)
            F[0:m_control] = - u_n[i, :].reshape(m_control, 1)
            F = np.array(F)
            F[0] = F[0] / um[0]
            F[-1] = - 10
            fx = dyn._f(state_i, params)
            gx = dyn._g(state_i, params)

            fx = fx.reshape(n_state, 1)
            gx = gx.reshape(n_state, m_control)

            V, Lg, Lf = LfLg_new(state_i, goal, fx, gx, sm, sl)

            A = torch.hstack((- Lg, - V))
            B = Lf

            assert not A.is_cuda

            G = scipy.sparse.csc.csc_matrix(A)
            h = - np.array(B).reshape(j_const, 1)
            # u = scipy.optimize.linprog(F, A_ub=G, b_ub=h)

            u = solve_qp(Q, F, G, h, solver="osqp")

            # print(u)

            if u is None:
                u = u_n[i, :].reshape(1, m_control)
                # u = u.reshape(1,m_control)
            u = u[0:m_control]
            # print(u)
            u_nominal[i, :] = torch.tensor(u).reshape(1, m_control)
            # print(t.toc())
        return u_nominal

    def nominal_controller_batch(self, state, goal, u_n, dyn):
        """
        args:
            state (n_state,)
            goal (n_state,)
        returns:
            u_nominal (m_control,)
        """
        # state = state.cuda()
        # goal = torch.tensor(goal).cuda()
        # u_n = u_n.cuda()

        um, ul = self.dyn.control_limits()
        # um = torch.tensor(um).cuda()
        # ul = ul.cuda()
        n_state = self.n_state
        m_control = self.m_control
        params = self.params
        j_const = self.j_const

        batch_size = state.shape[0]

        # size_Q = (m_control + j_const) * batch_size

        # Q = csc_matrix(identity(size_Q))
        Q = torch.eye(m_control + j_const)  # .cuda()

        F = torch.ones(m_control + j_const, 1)  # .cuda()
        # A_l = torch.zeros(batch_size * j_const, size_Q)
        # B_l = torch.zeros(batch_size * j_const).reshape(batch_size * j_const, 1)
        u_n = u_n.reshape(batch_size, m_control)
        u_nominal = torch.zeros(batch_size, m_control)

        for i in range(batch_size):
            # t.tic()
            # print(i)
            # Q[m_control * i, m_control * i] = Q[m_control * i, m_control * i] / um[0]
            state_i = state[i, :].reshape(1, n_state)  # .cuda()
            F[0:m_control] = - u_n[i, :].reshape(m_control, 1)
            # F = np.array(F)
            F[0] = F[0] / um[0]
            F[-1] = - 10
            fx = dyn._f(state_i, params)
            gx = dyn._g(state_i, params)

            fx = fx.reshape(n_state, 1)
            gx = gx.reshape(n_state, m_control)

            V, Lg, Lf = LfLg_new(state_i, goal, fx, gx, 1, [np.pi / 8, -np.pi / 80])

            A = torch.hstack((- Lg, - V))

            # u = QPFunction(verbose=False)(Q, F.reshape(m_control+j_const), A, -Lf.reshape(j_const), torch.tensor([
            # ]).cuda(), torch.tensor([]).cuda())

            # A_l[i * j_const: (i+1) * j_const, i * (m_control + j_const): (i+1) * (m_control + j_const)] = A
            # # Lf = np.array(Lf)
            #
            # B_l[i * j_const: (i+1) * j_const] = Lf.reshape(j_const, 1)
            # G = A_l.cuda()  # scipy.sparse.csc.csc_matrix(A_l)
            # # F = np.array(F)
            # # h = - np.array(B_l)
            # h = - B_l.cuda()
            #
            u = solve_qp(Q, F, A, -Lf.reshape(j_const, 1), solver="osqp")
            # print(G.shape)
            # print(h.shape)
            #
            # # u = QPFunction(verbose=False)(Q, F, G, h, torch.tensor([]).cuda(), torch.tensor([]).cuda())
            #
            # print(u.shape)
            if u is None:
                u = u_n[i * m_control: (i + 1) * m_control].reshape(1, m_control)

            u = u.clone().cpu()
            u_nominal[i, :] = u[0][0:m_control].reshape(1, m_control)
        # for i in range(batch_size):
        #     u_nominal[i, :] = torch.tensor(u[i * m_control: (i + 1) * m_control]).reshape(1, m_control)
        # print(t.toc())

        return u_nominal

    def fault_controller(self, u_nominal, fx, gx, h, grad_h):
        """
        args:
            state (n_state,)
            goal (n_state,)
        returns:
            u_nominal (m_control,)
        """
        um, ul = self.dyn.control_limits()
        
        bs = u_nominal.shape[0]
        
        u_neural = u_nominal.clone()

        m_control = self.m_control

        size_Q = m_control + 1

        num_constraints = (m_control + 1) * 2 + 1

        Q = csc_matrix(identity(size_Q))

        Q = Q / 100

        A_all = torch.tensor([]).reshape(0, 0)
        B_all = torch.tensor([]).reshape(0, 1)
        F_all = torch.tensor([]).reshape(0, 1)

        for i in range(bs):
            u_nom = u_nominal[i, :].reshape(m_control)

            F = - torch.hstack((u_nom, torch.tensor(1.0))).reshape(size_Q, 1)
            
            F = F / 100

            F[-1] = -1

            Lg = torch.matmul(grad_h[i, :, :], gx[i, :, :]).detach()
            Lf = torch.matmul(grad_h[i, :, :], fx[i, :, :]).detach()

            if h[i] == 0:
                h[i] = 1e-4

        # noinspection PyTypeChecker
            A = torch.hstack((- Lg.reshape(1, m_control), -h[i].reshape(1, 1)))
            B = Lf.detach().cpu()

            lb = torch.vstack((ul.reshape(self.m_control, 1), torch.tensor(-100000000).reshape(1, 1)))
            ub = torch.vstack((um.reshape(self.m_control, 1), torch.tensor(100000000).reshape(1, 1)))
            A_in = torch.tensor(
                [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [-1, 0, 0, 0, 0],
                [0, -1, 0, 0, 0], [0, 0, -1, 0, 0], [0, 0, 0, -1, 0], [0, 0, 0, 0, -1]])

            A_mat = torch.vstack((A.clone().detach(), A_in))

            B_in = torch.vstack((ub, -lb))

            B_mat = torch.vstack((B, B_in.reshape(2 * self.m_control + 2, 1)))

            num_constraints = A_mat.shape[0]

            num_vars = A_mat.shape[1]

            current_row = A_all.shape[0]

            current_col = A_all.shape[1]

            A_all = torch.hstack((A_all, torch.zeros(current_row, num_vars)))

            A_mat = torch.hstack((torch.zeros(num_constraints, current_col), A_mat.clone()))

            A_all = torch.vstack((A_all, A_mat))
            
            B_all = torch.vstack((B_all, B_mat))

            F_all = torch.vstack((F_all, F))
            
        A_mat = scipy.sparse.csc.csc_matrix(A_mat)
        
        B_mat = np.array(B_mat)

        q_size = F_all.shape[0]
                
        Q_all = torch.eye(q_size)

        Q_mat = csc_matrix(Q_all)
        
        F_mat = np.array(F_all)

        A_mat = scipy.sparse.csc.csc_matrix(A_mat)

        B_mat = np.array(B_mat)

        # m.update(q=F_mat, u=B_mat)

        # m.update(Ax=A_mat.data)

        # res = m.solve()
        # u = res.x

        # if u is None:
        #     u_neural = u_nominal.clone()
        # else:
        #     u = torch.tensor(u).reshape(bs, m_control + 1).type_as(u_nominal)
        #     u_neural = u[:, :m_control]
        try:
            u = solve_qp(Q_mat, F_mat, A_mat, B_mat, solver="osqp")
            u = torch.tensor(u).reshape(bs, m_control + 1).type_as(u_nominal)
            u_neural = u[:, :m_control]
        except:
            u_neural = u_nominal.clone()
            
        return u_neural.reshape(bs, m_control)
    
    def fault_controller_batch(self, u_nominal, fx, gx, h, grad_h):
        """
        args:
            state (n_state,)
            goal (n_state,)
        returns:
            u_nominal (m_control,)
        """
        um, ul = self.dyn.control_limits()
        
        bs = u_nominal.shape[0]

        um = um.repeat(bs, 1)
        ul = ul.repeat(bs, 1)
        
        u_neural = u_nominal.clone()

        m_control = self.m_control

        size_Q = (m_control + 1) * bs

        Q = csc_matrix(identity(size_Q))

        Q = Q / 100

        # for i in range(bs):
        u_nom = u_nominal.reshape(bs, m_control)
        F = torch.hstack((u_nom, - torch.ones(bs, 1))).reshape(size_Q, 1)

        F = - np.array(F)
    # F[0] = F[0] / um[0]

        F = F / 100

        Lg = torch.matmul(grad_h, gx).detach()
        Lf = torch.matmul(grad_h, fx).detach()
        
        # if h[i] == 0:
        #     h[i] = 1e-4

    # noinspection PyTypeChecker
        A = torch.hstack((- Lg.reshape(bs, m_control), -h.reshape(bs, 1)))

        B = Lf.detach().cpu()
        
        lb = torch.vstack((ul.reshape(self.m_control, bs), -10000*torch.ones(1, bs)))
        ub = torch.vstack((um.reshape(self.m_control, bs), 1000*torch.ones(1, bs)))
        
        A_in = torch.tensor(
            [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [-1, 0, 0, 0, 0],
            [0, -1, 0, 0, 0], [0, 0, -1, 0, 0], [0, 0, 0, -1, 0], [0, 0, 0, 0, -1]])
        
        A_in = A_in.repeat(bs, 1)
        
        A = torch.vstack((A.clone().detach(), A_in))

        B_in = torch.vstack((ub, -lb))
        B = torch.vstack((B.reshape(B.shape[0], 1), B_in.reshape(2 * (self.m_control + 1) * bs, 1)))
    
        B = np.array(B)

        print(A.shape)
        
        A = scipy.sparse.csc.csc_matrix(A)

        try:
            u = solve_qp(Q, F, A, B, solver="osqp")
        except:
            u = None

        if u is None:
            u_neural = u_nominal
        else:
            # for j in range(m_control):
            #     if u[j] < ul[j]:
            #         u[j] = ul[j].clone()
            #     if u[j] > um[j]:
            #         u[j] = um[j].clone()
            u = torch.tensor(u).reshape(m_control + 1, bs)
            u = u[0:m_control, :]
            u_neural = u.reshape(bs, m_control)
            
            # u_neural = u[0:self.m_control].reshape(1, m_control)
        return u_neural.reshape(bs, m_control)
    
    def neural_controller(self, u_nominal, fx, gx, h, grad_h, fault_start):
        """
        args:
            state (n_state,)
            goal (n_state,)
        returns:
            u_nominal (m_control,)
        """
        um, ul = self.dyn.control_limits()

        m_control = self.m_control

        size_Q = m_control + 1

        Q = csc_matrix(identity(size_Q))
        # Q[0,0] = 1 / um[0]
        F = torch.hstack((torch.tensor(u_nominal).reshape(m_control), torch.tensor(1.0))).reshape(size_Q, 1)

        F = - np.array(F)
        # F[0] = F[0] / um[0]

        Q = Q / 100
        F = F / 100

        F[-1] = -1

        Lg = torch.matmul(grad_h, gx).detach()
        Lf = torch.matmul(grad_h, fx).detach()

        if fault_start == 1:
            # uin = um[self.fault_control_index] * (Lg[0, 0, self.fault_control_index] > 0) + \
            #       ul[self.fault_control_index] * (Lg[0, 0, self.fault_control_index] <= 0)
            # Lf = Lf - torch.abs(Lg[0, 0, self.fault_control_index]) * uin
            Lf = Lf - torch.abs(Lg[0, 0, self.fault_control_index]) * um[self.fault_control_index]
            Lg[0, 0, self.fault_control_index] = 0.0

        if h == 0:
            h = 1e-4

        # noinspection PyTypeChecker
        A = torch.hstack((- Lg.reshape(1, m_control), -h))
        B = Lf.detach().cpu()

        lb = torch.vstack((ul.reshape(self.m_control, 1), torch.tensor(-1000000).reshape(1, 1)))
        ub = torch.vstack((um.reshape(self.m_control, 1), torch.tensor(1000000).reshape(1, 1)))
        A_in = torch.vstack((torch.eye(m_control+1), -torch.eye(m_control + 1)))
        # A_in = torch.tensor(
        #     [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [-1, 0, 0, 0, 0],
        #      [0, -1, 0, 0, 0], [0, 0, -1, 0, 0], [0, 0, 0, -1, 0], [0, 0, 0, 0, -1]])

        A = torch.vstack((A.clone().detach(), A_in))

        B_in = torch.vstack((ub, -lb))
        B = torch.vstack((B, B_in.reshape(2 * self.m_control + 2, 1, 1)))

        B = np.array(B)

        # print(A)
        A = scipy.sparse.csc.csc_matrix(A)
        u = solve_qp(Q, F, A, B, solver="osqp")
        # , lb = lb.reshape(self.m_control + 1, 1, 1), ub = ub.reshape(self.m_control + 1, 1, 1),

        if u is None:
            u_neural = u_nominal.reshape(m_control)
        else:
            u_neural = torch.tensor([u[0:self.m_control]]).reshape(1, m_control)

        return u_neural
    
    def neural_controller_gamma(self, u_nominal, fx, gx, h, grad_h, fault_start, fault_index=-1):
        """
        args:
            state (n_state,)
            goal (n_state,)
        returns:
            u_nominal (m_control,)
        """
        um, ul = self.dyn.control_limits()

        m_control = self.m_control

        size_Q = m_control + 1

        Q = csc_matrix(identity(size_Q))
        # Q[0,0] = 1 / um[0]
        F = torch.hstack((u_nominal.reshape(m_control), torch.tensor(1.0))).reshape(size_Q, 1)

        F = - np.array(F)
        # F[0] = F[0] / um[0]

        Q = Q / 100
        F = F / 100

        F[-1] = -1

        Lg = torch.matmul(grad_h, gx).detach()
        Lf = torch.matmul(grad_h, fx).detach()

        if fault_start == 1 and fault_index >= 0:
            F[fault_index] = 0
            Q[fault_index, fault_index] = 100
            # uin = um[fault_index] * (Lg[0, 0, fault_index] > 0) + \
                #   ul[fault_index] * (Lg[0, 0, fault_index] <= 0)
            # Lf = Lf - torch.abs(Lg[0, 0, fault_index]) * uin
            Lf = Lf - torch.abs(Lg[0, 0, fault_index]) * um[fault_index]
            Lg[0, 0, fault_index] = 0.0

        if h == 0:
            h = 1e-4

        # noinspection PyTypeChecker
        A = torch.hstack((- Lg.reshape(1, m_control), -h))
        B = Lf.detach().cpu()

        
        lb = torch.vstack((ul.reshape(self.m_control, 1), torch.tensor(-1000000).reshape(1, 1)))
        ub = torch.vstack((um.reshape(self.m_control, 1), torch.tensor(1000000).reshape(1, 1)))
        A_in = torch.tensor(
            [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [-1, 0, 0, 0, 0],
             [0, -1, 0, 0, 0], [0, 0, -1, 0, 0], [0, 0, 0, -1, 0], [0, 0, 0, 0, -1]])

        A = torch.vstack((A.clone().detach(), A_in))

        B_in = torch.vstack((ub, -lb))
        B = torch.vstack((B, B_in.reshape(2 * self.m_control + 2, 1, 1)))

        B = np.array(B)
        
        if fault_index >= 0:
            B[fault_index + 1] = 0
            B[fault_index + m_control + 2]  = 0
    
        # print(A)
        A = scipy.sparse.csc.csc_matrix(A)
        u = solve_qp(Q, F, A, B, solver="osqp")
        # , lb = lb.reshape(self.m_control + 1, 1, 1), ub = ub.reshape(self.m_control + 1, 1, 1),

        if u is None:
            u_neural = u_nominal.reshape(m_control)
        else:
            u_neural = torch.tensor(np.array(u[0:self.m_control])).reshape(1, m_control)

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

        lo = sl.reshape(n_dims,)
        hi = sm.reshape(n_dims,)

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

    def doth_max(self, grad_h, fx, gx, um, ul):

        bs = grad_h.shape[0]

        doth = torch.matmul(grad_h, fx)
        # doth = doth.reshape(bs, 1)
        LhG = torch.matmul(grad_h, gx)

        sign_grad_h = torch.sign(LhG).reshape(bs, 1, self.m_control)

        if self.fault == 0:
            doth = doth + torch.matmul(sign_grad_h, um.reshape(bs, self.m_control, 1)) + \
                   torch.matmul(1 - sign_grad_h, ul.reshape(bs, self.m_control, 1))
        else:
            for i in range(self.m_control):
                if i == self.fault_control_index:
                    doth = doth.reshape(bs, 1) - sign_grad_h[:, 0, i].reshape(bs, 1) * um[:, i].reshape(bs, 1) - \
                           (1 - sign_grad_h[:, 0, i].reshape(bs, 1)) * ul[:, i].reshape(bs, 1)
                else:
                    doth = doth.reshape(bs, 1) + sign_grad_h[:, 0, i].reshape(bs, 1) * um[:, i].reshape(bs, 1) + \
                           (1 - sign_grad_h[:, 0, i].reshape(bs, 1)) * ul[:, i].reshape(bs, 1)

        return doth.reshape(1, bs)

    def doth_max_alpha(self, h, grad_h, fx, gx, um, ul):

        bs = grad_h.shape[0]

        doth = torch.matmul(grad_h, fx)
        # doth = doth.reshape(bs, 1)
        LhG = torch.matmul(grad_h, gx).reshape(bs, self.m_control)

        LhG = torch.hstack((LhG, h))
        vec_ones = 10 * torch.ones(bs, 1)
        # noinspection PyTypeChecker
        um = torch.hstack((um, vec_ones)).reshape(self.m_control + 1, bs)
        # noinspection PyTypeChecker
        ul = torch.hstack((ul, -1 * vec_ones)).reshape(self.m_control + 1, bs)

        sign_grad_h = torch.sign(LhG).reshape(bs, 1, self.m_control + 1)
        # ind_pos = sign_grad_h > 0
        # ind_neg = sign_grad_h <= 0

        uin = um.reshape(self.m_control + 1, bs) * \
              (sign_grad_h > 0).reshape(self.m_control + 1, bs) - \
              ul.reshape(self.m_control + 1, bs) * \
              (sign_grad_h <= 0).reshape(self.m_control + 1, bs)

        doth = doth + torch.matmul(torch.abs(LhG).reshape(bs, 1, self.m_control + 1),
                                   uin.reshape(bs, self.m_control + 1, 1))
        if self.fault == 1:
            doth = doth.reshape(bs, 1) - 1.5 * torch.abs(LhG[:, self.fault_control_index]).reshape(bs, 1) * uin[
                                                                                                            self.fault_control_index,
                                                                                                            :].reshape(
                bs, 1)

        return doth.reshape(1, bs)
