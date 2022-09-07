import torch
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
                 j_const = 1,
                 dt=0.05, 
                 fault = 0,
                 fault_control_index = -1):

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

        for j in range(self.m_control):
            if self.fault == 1 and self.fault_control_index == j:
                u[:,j] = u[:,j].clone().detach().reshape(batch_size,1)
            else:
                u[:,j] = u[:,j].clone().detach().requires_grad_(True).reshape(batch_size,1)
        
        dsdt = fx + torch.matmul(gx,u)

        return dsdt

    
    def nominal_controller(self, state, goal, u_n, dyn,constraints):
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

        size_Q = m_control + j_const

        Q = csc_matrix(10*identity(size_Q))

        F = torch.ones(size_Q,1)
        
        F[0:m_control] = - 10 * u_n.reshape(m_control,1)
        F = np.array(F)

        fx = dyn._f(state,params)
        gx = dyn._g(state,params)

        fx = fx.reshape(n_state,1)
        gx = gx.reshape(n_state,m_control)

        V, Lg, Lf = constraints.LfLg_new(state,goal,fx,gx,n_state, m_control, j_const, 1, [sm[1], sl[1]])

        A = torch.hstack((Lg, V))
        B = Lf

        G = scipy.sparse.csc.csc_matrix(A)
        h = np.array(B)
        u = solve_qp(Q, F, G, h, solver="osqp")

        if (u is None):
            u = np.array(um) / 2
            u = u.reshape(1,m_control)

        u_nominal = torch.tensor([u[0:self.m_control]]).reshape(1,m_control)

        return u_nominal


    def neural_controller(self, u_nominal, fx, gx, h, grad_h):
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

        size_Q = m_control + 1

        Q = csc_matrix(10*identity(size_Q))
        F = torch.hstack((torch.tensor(u_nominal).reshape(m_control), torch.tensor(1.0))).reshape(size_Q,1)

        F = np.array(F)

        Lg = torch.matmul(grad_h, gx)
        Lf = torch.matmul(grad_h, fx)
        
        A = torch.hstack((Lg.reshape(1,m_control), h))
        A = torch.tensor(A.detach().cpu())
        B = Lf.detach().cpu().numpy() 
        B = np.array(B)

        A = scipy.sparse.csc.csc_matrix(A)
        u = solve_qp(Q, F, A, B, solver="osqp")

        if (u is None):
            u = np.array(um) / 2
            u = u.reshape(1,m_control)

        u_neural = torch.tensor([u[0:self.m_control]]).reshape(1,m_control)

        return u_neural