"""Define a dynamical system for a 3D quadrotor"""
from typing import Tuple, List, Optional, Callable

import torch
import numpy as np

from torch.autograd.functional import jacobian

from .control_affine_system_new import ControlAffineSystemNew

from .utils import (
    lqr,
    continuous_lyap,
)
# from .utils import grav, Scenario, ScenarioList
# from neural_clbf.systems.utils import grav, Scenario, ScenarioList


class DI(ControlAffineSystemNew):
    """
    Represents a planar quadrotor.

    The system has state

        x = [px, py, pz, vx, vy, vz, phi, theta, psi]

    representing the position, orientation, and velocities of the quadrotor, and it
    has control inputs

        u = [f, phi_dot, theta_dot, psi_dot]

    The system is parameterized by
        m: mass

    NOTE: Z is defined as positive downwards
    """

    def __init__(
            self,
            x: torch.Tensor,
            dim,
            nominal_parameters,
            goal: torch.Tensor,
            dt: float = 0.01,
            
    ):
        self.x = x
        self.dim = dim
        self.fault = 0
        self.goal = goal
        self.dt = dt
        self.controller_dt = dt
        self.nominal_params = None

        """
        Initialize the quadrotor.

        args:
            nominal_params: a dictionary giving the parameter values for the system.
                            Requires keys ["m"]
            dt: the timestep to use for the simulation
            controller_dt: the timestep for the LQR discretization. Defaults to dt
        raises:
            ValueError if nominal_params are not valid for this system
        """
        super().__init__(x, goal, dt)
    

    @property
    def angle_dims(self) -> None:
        return None

    def validate_params(self, params) -> bool:

        # print(params)
        """Check if a given set of parameters is valid

        args:
            params: a dictionary giving the parameter values for the system.
                    Requires keys ["m"]
        returns:
            True if parameters are valid, False otherwise
        """
        valid = True
        # Make sure all needed parameters were provided

        return valid

    @property
    def n_dims(self) -> int:
        return self.dim * 2


    @property
    def n_controls(self) -> int:
        return self.dim
    
    def compute_linearized_controller(self, scenarios=None):
        """
        Computes the linearized controller K and lyapunov matrix P.
        """
        # We need to compute the LQR closed-loop linear dynamics for each scenario
        Acl_list = []
        # Default to the nominal scenario if none are provided
        # if scenarios is None:
        #     scenarios = [None]

        # # For each scenario, get the LQR gain and closed-loop linearization
        # for s in scenarios:
            # Compute the LQR gain matrix for the nominal parameters
        Act, Bct = self.linearized_ct_dynamics_matrices()
        A, B = self.linearized_dt_dynamics_matrices()

        # Define cost matrices as identity
        Q = np.eye(self.n_dims)
        R = np.eye(self.n_controls)

        # Get feedback matrix
        K_np = lqr(A, B, Q, R)
        self.K = torch.tensor(K_np)

        Acl_list.append(Act - Bct @ K_np)

        # If more than one scenario is provided...
        # get the Lyapunov matrix by robustly solving Lyapunov inequalities

        self.P = torch.tensor(continuous_lyap(Acl_list[0], Q))

    # @property
    def state_limits(self):
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upperx = 100 * torch.ones(int(self.n_dims / 2)).reshape(1, int(self.n_dims / 2))
        upperv = 10 * torch.ones(int(self.n_dims / 2)).reshape(1, int(self.n_dims / 2))
        upper_limit = torch.hstack((upperx, upperv)).reshape(self.n_dims,1)

        lower_limit = -1.0 * upper_limit

        # lower_limit = torch.tensor(lower_limit)
        # upper_limit = torch.tensor(upper_limit)

        return upper_limit, lower_limit

    # @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control
        limits for this system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.tensor([10, 10, 10])
        lower_limit = -1.0 * upper_limit

        # lower_limit = torch.tensor(lower_limit)
        # upper_limit = torch.tensor(upper_limit)

        return upper_limit, lower_limit

    def safe_mask(self, x, fault=0):
        """Return the mask of x indicating safe regions for the obstacle task

        args:
            x: a tensor of points in the state space
        """
        # fault = self.fault
        if fault == 0:
            safe_x = 50.0
            safe_u = 7.0
            safe_x_l = - 50.0
            safe_u_l = -7.0
        else:
            safe_x = 60.0
            safe_u = 8.0
            safe_x_l = - 60.0
            safe_u_l = -8.0
        
        safe_mask = torch.ones_like(x[:, 0], dtype=torch.bool)
        for j in range(self.dim):
            safe_maskx = torch.logical_and(
                x[:, j] <= safe_x, x[:, j] >= safe_x_l)
            safe_mask = torch.logical_and(safe_mask, safe_maskx)
        
        for j in range(self.dim):
            safe_masku = torch.logical_and(
                x[:, j + self.dim] <= safe_u, x[:, j + self.dim] >= safe_u_l)
            safe_mask = torch.logical_and(safe_mask, safe_masku)
        # safe_mask2 = torch.logical_and(
        #     x[:, DI.Y] <= safe_x, x[:, DI.Y] >= safe_x_l)
        # safe_mask3 = torch.logical_and(
        #     x[:, DI.Z] <= safe_x, x[:, DI.Z] >= safe_x_l)
        # safe_mask = torch.logical_and(safe_mask, safe_mask2)
        # safe_mask = torch.logical_and(safe_mask, safe_mask3)

        return safe_mask

    def safe_limits(self, sm=[], sl=[]):
        """Return the mask of x indicating safe regions for the obstacle task

        args:
            x: a tensor of points in the state space
        """
        fault = self.fault
        if fault == 0:
            safe_x = 50.0
            safe_u = 7.0
            safe_x_l = - 50.0
            safe_u_l = -7.0
        else:
            safe_x = 60.0
            safe_u = 8
            safe_x_l = - 60.0
            safe_u_l = -8
        # safe_radius = 3

        safe_l = torch.vstack((safe_x_l * torch.ones(self.dim,1), safe_u_l * torch.ones(self.dim,1)))
        safe_m = torch.vstack((safe_x * torch.ones(self.dim,1), safe_u * torch.ones(self.dim,1)))

        # safe_mask = torch.logical_and(safe_mask, x[:, FixedWing.BETA] <= safe_beta)
        # safe_mask = torch.logical_and(safe_mask, x[:, FixedWing.BETA] >= -safe_beta)

        return safe_m, safe_l

    def unsafe_mask(self, x, fault=0):
        """Return the mask of x indicating unsafe regions for the obstacle task

        args:
            x: a tensor of points in the state space
        """
        # fault = self.fault
        if fault == 0:
            unsafe_x = 60.0
            unsafe_u = 7.5
            unsafe_x_l = - 60.0
            unsafe_u_l = -7.5
        else:
            unsafe_x = 70.0
            unsafe_u = 8.5
            unsafe_x_l = - 70.0
            unsafe_u_l = -8.5

        unsafe_mask = torch.zeros_like(x[:, 0], dtype=torch.bool)

        for j in range(self.dim):
            unsafe_maskx = torch.logical_or(
                x[:, j] >= unsafe_x, x[:, j] <= unsafe_x_l)
            unsafe_mask = torch.logical_or(unsafe_mask, unsafe_maskx)

        for j in range(self.dim):
            unsafe_masku = torch.logical_or(
                x[:, j + self.dim] >= unsafe_u, x[:, j + self.dim] <= unsafe_u_l)
            unsafe_mask = torch.logical_or(unsafe_mask, unsafe_masku)
        # unsafe_mask2 = torch.logical_and(
        #     x[:, :, DI.Y] >= unsafe_x, x[:, :, DI.Y] <= unsafe_x_l)
        # unsafe_mask3 = torch.logical_and(
        #     x[:, :, DI.Z] >= unsafe_x, x[:, :, DI.Z] <= unsafe_x_l)
        # unsafe_mask = torch.logical_and(unsafe_mask, unsafe_mask2)
        # unsafe_mask = torch.logical_and(unsafe_mask, unsafe_mask3)

        return unsafe_mask

    def mid_mask(self, x, fault=0):
        mid_mask =   (~ self.safe_mask(x, fault)) * (~ self.unsafe_mask(x, fault))
        return mid_mask
    # def multi_unsafe_mask(self, x1, x2):
    #     """Return the mask of x indicating safe regions for the obstacle task

    #     args:
    #         x: a tensor of points in the state space
    #     """
    #     fault = self.fault
    #     if fault == 0:
    #         unsafe_dist = 1.0
    #     else:
    #         unsafe_dist = 2.0
    #     x1pos = torch.tensor([x1[:, DI.X], x1[:, DI.Y], x1[:, DI.Z]])
    #     x2pos = torch.tensor([x2[:, DI.X], x2[:, DI.Y], x2[:, DI.Z]])
    #     unsafe_mask = torch.norm(x1pos - x2pos) <= unsafe_dist

        # return unsafe_mask

    def goal_mask(self, x, xg):
        """Return the mask of x indicating points in the goal set (within 0.2 m of the
        goal).

        args:
            x: a tensor of points in the state space
        """
        goal_mask = torch.ones_like(x[:, :, 0], dtype=torch.bool)

        # Define the goal region as being near the goal

        near_goal = torch.linalg.norm(x - xg) <= 0.3
        goal_mask.logical_and_(near_goal)

        # The goal set has to be a subset of the safe set
        goal_mask.logical_and_(self.safe_mask(x))

        return goal_mask

    def _f(self, x: torch.Tensor, params=None):
        """
        Return the control-independent part of the control-affine dynamics.

        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            f: bs x self.n_dims x 1 tensor
        """
        # Extract batch size and set up a tensor for holding the result
        
        batch_size = x.shape[0]
        x = x.reshape(batch_size, self.n_dims)
        # px = x[:, DI.X]
        # py = x[:, DI.Y]
        # pz = x[:, DI.Z]
        
        # ux = x[:, DI.U].reshape(batch_size, 1)
        # uy = x[:, DI.V].reshape(batch_size, 1)
        # uz = x[:, DI.W].reshape(batch_size, 1)

        f = torch.zeros((batch_size, self.n_dims, 1))
        f = f.type_as(x)
        for j in range(self.dim):
            f[:, j, :] = x[:, j + self.dim].reshape(batch_size, 1)
            f[:, j, :] += 0.1 * x[:, np.mod(j + 1, self.dim)].reshape(batch_size, 1) * x[:, np.mod(j + 2, self.dim)].reshape(batch_size, 1)
            # f[:, :, DI.Y] = uy
            # f[:, :, DI.Z] = uz

        return f

    def _g(self, x: torch.Tensor, params=None):
        """
        Return the control-independent part of the control-affine dynamics.

        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            g: bs x self.n_dims x self.n_controls tensor
        """
        # Extract batch size and set up a tensor for holding the result
        # print(x.shape)
        # x = torch.tensor(x)
        batch_size = x.shape[0]

        
        g = torch.zeros((batch_size, self.n_dims, self.n_controls))
        g = g.type_as(x)

        # Extract the needed parameters
        for j in range(self.dim):
            g[:, j + self.dim, j] = torch.ones(batch_size,)
        # g[:, DI.V, DI.AY] = torch.ones(batch_size,)
        # g[:, DI.W, DI.AZ] = torch.ones(batch_size,)

        return g

    def u_in(self):
        u_eq = torch.zeros((1, self.n_controls))

        return u_eq

    # @property
    def u_eq(self):
        u_eq = torch.zeros((1, self.n_controls))

        return u_eq

    def sample_state_space(self, num_samples: int) -> torch.Tensor:
        """Sample uniformly from the state space"""
        x_max, x_min = self.state_limits()

        # Sample uniformly from 0 to 1 and then shift and scale to match state limits
        x = torch.Tensor(num_samples, self.n_dims).uniform_(0.0, 1.0)
        for i in range(self.n_dims):
            x[:, i] = x[:, i] * (x_max[i] - x_min[i]) + x_min[i]

        return x

    def sample_with_mask(
        self,
        num_samples: int,
        mask_fn: Callable[[torch.Tensor], torch.Tensor],
        max_tries: int = 5000,
    ) -> torch.Tensor:
        """Sample num_samples so that mask_fn is True for all samples. Makes a
        best-effort attempt, but gives up after max_tries, so may return some points
        for which the mask is False, so watch out!
        """
        # Get a uniform sampling
        samples = self.sample_state_space(num_samples)

        # While the mask is violated, get violators and replace them
        # (give up after so many tries)
        for _ in range(max_tries):
            violations = torch.logical_not(mask_fn(samples))
            if not violations.any():
                break

            new_samples = int(violations.sum().item())
            samples[violations] = self.sample_state_space(new_samples)

        return samples

    def sample_safe(self, num_samples: int, max_tries: int = 5000) -> torch.Tensor:
        """Sample uniformly from the safe space. May return some points that are not
        safe, so watch out (only a best-effort sampling).
        """
        return self.sample_with_mask(num_samples, self.safe_mask, max_tries)

    def sample_unsafe(self, num_samples: int, max_tries: int = 15000) -> torch.Tensor:
        """Sample uniformly from the unsafe space. May return some points that are not
        unsafe, so watch out (only a best-effort sampling).
        """
        return self.sample_with_mask(num_samples, self.unsafe_mask, max_tries)

    def sample_mid(self, num_samples: int, max_tries: int = 15000) -> torch.Tensor:
        """Sample uniformly from the unsafe space. May return some points that are not
        unsafe, so watch out (only a best-effort sampling).
        """
        return self.sample_with_mask(num_samples, self.mid_mask, max_tries)

    # @property
    # def n_dims(self) -> int:
    #     return DI.N_DIMS

    # @property
    # def n_controls(self) -> int:
    #     return DI.N_CONTROLS

    def compute_AB_matrices(self, state, u):
        """Compute the linearized continuous-time state-state derivative transfer matrix
        about the goal point"""
        # Linearize the system about the x = 0, u = 0
        # if scenario is None:
        scenario = self.nominal_params

        x0 = state
        u0 = u
        dynamics = lambda x0: self.closed_loop_dynamics(x0, u0, scenario).squeeze()
        A = jacobian(dynamics, x0).squeeze().cpu().numpy()
        A = np.reshape(A, (self.n_dims, self.n_dims))

        return A

    def EKF_gain(self, A, C, P):
        A_EKF = A

        Q = np.eye(self.n_dims + self.n_controls) * 0.1
        
        R = np.eye(self.n_dims) / 100
        
        P = A @ P @ A.T + Q
        S = C @ P @ C.T + R
        K = P @ C.T @ np.linalg.inv(S)
        P = (np.eye(self.n_dims + self.n_controls) - K @ C) @ P
        # B_EKF = np.transpose(C)
        # L = lqr(A_EKF, B_EKF, Q, R)
        # L = np.transpose(L)
    
        return torch.tensor(K, dtype=torch.float32), P

    def compute_A_matrix(self, scenario) -> np.ndarray:
        """Compute the linearized continuous-time state-state derivative transfer matrix
        about the goal point"""
        # Linearize the system about the x = 0, u = 0
        if scenario is None:
            scenario = self.nominal_params

        x0 = self.goal_point
        u0 = self.u_eq()
        dynamics = lambda x: self.closed_loop_dynamics(x, u0, scenario).squeeze()
        A = jacobian(dynamics, x0).squeeze().cpu().numpy()
        A = np.reshape(A, (self.n_dims, self.n_dims))

        return A

    def compute_B_matrix(self, scenario) -> np.ndarray:
        """Compute the linearized continuous-time state-state derivative transfer matrix
        about the goal point"""
        if scenario is None:
            scenario = self.nominal_params

        # Linearize the system about the x = 0, u = 0
        B = self._g(self.goal_point, scenario).squeeze().cpu().numpy()
        B = np.reshape(B, (self.n_dims, self.n_controls))

        return B

    def linearized_ct_dynamics_matrices(
        self, scenario=None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the continuous time linear dynamics matrices, dx/dt = Ax + Bu"""
        A = self.compute_A_matrix(scenario)
        B = self.compute_B_matrix(scenario)

        return A, B

    def linearized_dt_dynamics_matrices(
        self, scenario=None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the continuous time linear dynamics matrices, x_{t+1} = Ax_{t} + Bu
        """
        Act, Bct = self.linearized_ct_dynamics_matrices(scenario)
        A = np.eye(self.n_dims) + self.controller_dt * Act
        B = self.controller_dt * Bct

        return A, B

    

    def control_affine_dynamics(
        self, x: torch.Tensor, params
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (f, g) representing the system dynamics in control-affine form:

            dx/dt = f(x) + g(x) u

        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            f: bs x self.n_dims x 1 tensor representing the control-independent dynamics
            g: bs x self.n_dims x self.n_controls tensor representing the control-
               dependent dynamics
        """
        # Sanity check on input
        assert x.ndim == 2
        assert x.shape[1] == self.n_dims

        # If no params required, use nominal params
        if params is None:
            params = self.nominal_params

        return self._f(x, params), self._g(x, params)

    def closed_loop_dynamics(
        self, x: torch.Tensor, u: torch.Tensor, params
    ) -> torch.Tensor:
        """
        Return the state derivatives at state x and control input u

            dx/dt = f(x) + g(x) u

        args:
            x: bs x self.n_dims tensor of state
            u: bs x self.n_controls tensor of controls
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            xdot: bs x self.n_dims tensor of time derivatives of x
        """
        # Get the control-affine dynamics
        # print(x.ndim)
        f, g = self.control_affine_dynamics(x, params=params)
        # print(f)

        # print(g)
        # Compute state derivatives using control-affine form
        xdot = f + torch.bmm(g, u.unsqueeze(-1)).reshape(f.shape)

        return xdot.view(x.shape)

    def u_nominal(self, x: torch.Tensor, op_point=None) -> torch.Tensor:
        """
        Compute the nominal control for the nominal parameters, using LQR unless
        overridden

        args:
            x: bs x self.n_dims tensor of state
        returns:
            u_nominal: bs x self.n_controls tensor of controls
        """
        # Compute nominal control from feedback + equilibrium control
        K = self.K.type_as(x)
        if op_point is None:
            op_point = self.goal

        goal = op_point.squeeze().type_as(x)
        u_nominal = -(K @ (x - goal).T).T

        # Adjust for the equilibrium setpoint
        u = u_nominal + self.u_eq().type_as(x)

        # Clamp given the control limits
        upper_u_lim, lower_u_lim = self.control_limits()
        for dim_idx in range(self.n_controls):
            u[:, dim_idx] = torch.clamp(
                u[:, dim_idx],
                min=lower_u_lim[dim_idx].item(),
                max=upper_u_lim[dim_idx].item(),
            )

        return u