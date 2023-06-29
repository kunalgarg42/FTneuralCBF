"""Define a dynamical system for a Crazyflie"""
from typing import Tuple, List, Optional, Callable

import torch
import numpy as np

from torch.autograd.functional import jacobian

from .control_affine_system_new import ControlAffineSystemNew

from .utils import (
    lqr,
    continuous_lyap,
)


class CrazyFlies(ControlAffineSystemNew):
    """
    The system has state
        x = [x, y, z, psi, theta, phi, u, v, w, r, q, p]
    representing the position, orientation, linear velocities, and angular velocities of the drone, and it
    has control inputs as the individual motor thrusts:
        u = [f1, f2, f3, f4] = T^(-1) * [U_1, U_2, U_3, U_4]
    where the transformation matrix is:
        T = [ [1, 1, 1, 1]
              [-d*sqrt(2), -d*sqrt(2), d*sqrt(2), d*sqrt(2)]
              [-d*sqrt(2), d*sqrt(2), d*sqrt(2), -d*sqrt(2)]
              [-CD/CT, CD/CT, -CD/CT, CD/CT] ]
    The system is parameterized by
        m: mass
        Ixx, Iyy, Izz: moments of inertia
        CT: thrust coefficient
        CD: drag coefficient
        d: quadcopter arm length
    NOTE: Z is defined as positive downwards
    """

    # Number of states and controls
    N_DIMS = 12
    N_CONTROLS = 4

    # State indices
    X = 0
    Y = 1
    Z = 2

    PSI = 3
    THETA = 4
    PHI = 5

    U = 6
    V = 7
    W = 8

    R = 9
    Q = 10
    P = 11

    # Control indices
    F_1 = 0
    F_2 = 1
    F_3 = 2
    F_4 = 3

    def __init__(
            self,
            x: torch.Tensor,
            nominal_params,
            goal: torch.Tensor,
            dt: float = 0.001,
    ):
        self.x = x
        self.goal = goal
        self.controller_dt = dt
        self.params = nominal_params
        """
        Initialize the quadrotor.
        args:
            nominal_params: a dictionary giving the parameter values for the system.
                            Requires keys ["m", "Ixx", "Iyy", "Izz", "CT", "CD", "d"]
            dt: the timestep to use for the simulation
            controller_dt: the timestep for the LQR discretization. Defaults to dt
        raises:
            ValueError if nominal_params are not valid for this system
        """
        super().__init__(x, nominal_params, goal, dt)

    def validate_params(self, params) -> bool:
        """Check if a given set of parameters is valid
        args:
            params: a dictionary giving the parameter values for the system.
                    Requires keys ["m", "Ixx", "Iyy", "Izz", "CT", "CD", "d"]
        returns:
            True if parameters are valid, False otherwise
        """
        valid = True
        # Make sure all needed parameters were provided
        valid = valid and "m" in params
        valid = valid and "Ixx" in params
        valid = valid and "Iyy" in params
        valid = valid and "Izz" in params
        valid = valid and "CT" in params
        valid = valid and "CD" in params
        valid = valid and "d" in params

        # Make sure all parameters are physically valid
        valid = valid and params["m"] > 0
        valid = valid and params["Ixx"] > 0
        valid = valid and params["Iyy"] > 0
        valid = valid and params["Izz"] > 0
        valid = valid and params["CT"] > 0
        valid = valid and params["CD"] > 0
        valid = valid and params["d"] > 0

        return valid

    @property
    def n_dims(self) -> int:
        return CrazyFlies.N_DIMS

    @property
    def angle_dims(self) -> List[int]:
        return [CrazyFlies.PSI, CrazyFlies.THETA, CrazyFlies.PHI]

    @property
    def n_controls(self) -> int:
        return CrazyFlies.N_CONTROLS

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

    def compute_linearized_controller(self, scenarios=None):
        """
        Computes the linearized controller K and lyapunov matrix P.
        """
        # We need to compute the LQR closed-loop linear dynamics for each scenario
        Acl_list = []
        # Default to the nominal scenario if none are provided
        if scenarios is None:
            scenarios = [self.nominal_params]

        # For each scenario, get the LQR gain and closed-loop linearization
        for s in scenarios:
            # Compute the LQR gain matrix for the nominal parameters
            Act, Bct = self.linearized_ct_dynamics_matrices(s)
            A, B = self.linearized_dt_dynamics_matrices(s)

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
        xdot = f + torch.bmm(g, u.unsqueeze(-1))
        return xdot.view(x.shape)

    # @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.ones(self.n_dims)
        upper_limit[CrazyFlies.X] = 15.0
        upper_limit[CrazyFlies.Y] = 15.0
        upper_limit[CrazyFlies.Z] = 15.0
        upper_limit[CrazyFlies.U] = 10.0
        upper_limit[CrazyFlies.V] = 10.0
        upper_limit[CrazyFlies.W] = 10.0
        upper_limit[CrazyFlies.PSI] = np.pi
        upper_limit[CrazyFlies.THETA] = np.pi / 3
        upper_limit[CrazyFlies.PHI] = np.pi / 3
        upper_limit[CrazyFlies.R] = 2
        upper_limit[CrazyFlies.Q] = 2
        upper_limit[CrazyFlies.P] = 2

        lower_limit = -1.0 * upper_limit
        lower_limit[CrazyFlies.Z] = -5

        return (upper_limit, lower_limit)

    def safe_limits(self, sm, sl, fault=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        # define upper and lower limits based around the nominal equilibrium input
        # if fault is None:
        #     params = self.params
        #     fault = params["fault"]

        # if fault == 0:
        safe_z_l = 0.5
        safe_z_u = 12.0
        safe_w_u = 8.0
        safe_w_l = -8.0
            # safe_angle = np.pi / 5.0
        # else:
        #     safe_z_l = 1.9
        #     safe_z_u = 12.1
        #     safe_w_u = 8.1
        #     safe_w_l = -8.1
            # safe_angle = np.pi / 4.8

        upper_limit = 0.9 * sm
        lower_limit = 0.9 * sl
        upper_limit[CrazyFlies.Z] = safe_z_u
        lower_limit[CrazyFlies.Z] = safe_z_l
        upper_limit[CrazyFlies.W] = safe_w_u
        lower_limit[CrazyFlies.W] = safe_w_l

        upper_limit[CrazyFlies.PSI] = np.pi
        lower_limit[CrazyFlies.PSI] = -np.pi

        return (upper_limit, lower_limit)

    # @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control
        limits for this system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.tensor([1, 1, 1, 1]) * 0.15
        lower_limit = 0.01 * upper_limit


        return (upper_limit, lower_limit)

    def safe_mask(self, x, fault=None):
        """Return the mask of x indicating safe regions for the task
        We don't want to crash to the floor, so we need a safe height to avoid the floor
        args:
            x: a tensor of points in the state space
        """
        # if fault is None:
        #     params = self.params
        #     fault = params["fault"]

        # if fault == 0:
        safe_z_l = 0.5
        safe_z_u = 12.0
        safe_w_u = 8.0
        safe_w_l = -8.0
        # else:
        #     safe_z_l = 1.9
        #     safe_z_u = 12.1
        #     safe_w_u = 8.1
        #     safe_w_l = -8.1

        safe_mask = torch.logical_and(x[:, CrazyFlies.Z] >= safe_z_l, x[:, CrazyFlies.Z] <= safe_z_u)
        safe_mask.logical_and_(x[:, CrazyFlies.W] >= safe_w_l)
        safe_mask.logical_and_(x[:, CrazyFlies.W] <= safe_w_u)

        return safe_mask

    def unsafe_mask(self, x):
        """Return the mask of x indicating unsafe regions for the task
        args:
            x: a tensor of points in the state space
        """
        unsafe_z_l = 0.1
        unsafe_z_u = 12.5
        unsafe_w_l = -9.0
        unsafe_w_u = 9.0

        unsafe_mask = torch.logical_or(x[:, CrazyFlies.Z] <= unsafe_z_l, x[:, CrazyFlies.Z] >= unsafe_z_u)
        unsafe_mask.logical_or_(x[:, CrazyFlies.W] >= unsafe_w_u)
        unsafe_mask.logical_or_(x[:, CrazyFlies.W] <= unsafe_w_l)

        return unsafe_mask

    def goal_mask(self, x):
        """Return the mask of x indicating points in the goal set (within 0.2 m of the
        goal).
        args:
            x: a tensor of points in the state space
        """
        goal_mask = torch.ones_like(x[:, 0], dtype=torch.bool)

        # Define the goal region as being near the goal
        near_goal = x.norm(dim=-1) <= 0.2
        goal_mask.logical_and_(near_goal)

        # The goal set has to be a subset of the safe set
        goal_mask.logical_and_(self.safe_mask(x))

        return goal_mask

    def _f(self, x: torch.Tensor, params):
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
        f = torch.zeros((batch_size, self.n_dims, 1)).type_as(x).to(x.device)
        

        # Extract the needed parameters
        m, Ixx, Iyy, Izz, CT, CD, d = params["m"], params["Ixx"], params["Iyy"], params["Izz"], params["CT"], params[
            "CD"], params["d"]

        # Extract state variables
        psi = x[:, CrazyFlies.PSI].reshape(batch_size, 1)
        theta = x[:, CrazyFlies.THETA].reshape(batch_size, 1)
        phi = x[:, CrazyFlies.PHI].reshape(batch_size, 1)

        u = x[:, CrazyFlies.U].reshape(batch_size, 1)
        v = x[:, CrazyFlies.V].reshape(batch_size, 1)
        w = x[:, CrazyFlies.W].reshape(batch_size, 1)

        r = x[:, CrazyFlies.R].reshape(batch_size, 1)
        q = x[:, CrazyFlies.Q].reshape(batch_size, 1)
        p = x[:, CrazyFlies.P].reshape(batch_size, 1)

        s_psi = torch.sin(psi).reshape(batch_size, 1)
        s_phi = torch.sin(phi).reshape(batch_size, 1)
        s_the = torch.sin(theta).reshape(batch_size, 1)

        c_psi = torch.cos(psi).reshape(batch_size, 1)
        c_phi = torch.cos(phi).reshape(batch_size, 1)
        c_the = torch.cos(theta).reshape(batch_size, 1)

        t_psi = torch.tan(psi).reshape(batch_size, 1)
        t_phi = torch.tan(phi).reshape(batch_size, 1)
        t_the = torch.tan(theta).reshape(batch_size, 1)

        # Derivatives of positions
        f[:, CrazyFlies.X] = w * (s_psi * s_phi + c_psi * c_phi * s_the) \
                             - v * (s_psi * c_phi - c_psi * s_phi * s_the) \
                             + u * c_psi * c_the
        f[:, CrazyFlies.Y] = v * (c_psi * c_phi + s_psi * s_phi * s_the) \
                             - w * (c_psi * s_phi - s_psi * c_phi * s_the) \
                             + u * s_psi * c_the
        f[:, CrazyFlies.Z] = w * c_psi * c_phi - u * s_the + v * s_phi * c_the

        # Derivatives of angles
        f[:, CrazyFlies.PSI] = p + r * c_phi * t_the + q * s_phi * t_the
        f[:, CrazyFlies.THETA] = q * c_phi - r * s_phi
        f[:, CrazyFlies.PHI] = r * c_phi / c_the + q * s_phi / c_the

        # Derivatives of linear velocities
        f[:, CrazyFlies.U] = r * v - q * w + 9.81 * s_the
        f[:, CrazyFlies.V] = p * w - r * u - 9.81 * s_phi * c_the
        f[:, CrazyFlies.W] = q * u - p * v - 9.81 * c_phi * c_the

        # Derivatives of angular velocities
        f[:, CrazyFlies.R] = (Ixx - Iyy) / Izz * p * q
        f[:, CrazyFlies.Q] = (Izz - Ixx) / Iyy * p * r
        f[:, CrazyFlies.P] = (Izz - Iyy) / Ixx * p * q

        return f

    def _g(self, x: torch.Tensor, params):
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
        batch_size = x.shape[0]
        g = torch.zeros((batch_size, self.n_dims, self.n_controls)).type_as(x).to(x.device)

        # Extract the needed parameters
        m, Ixx, Iyy, Izz, CT, CD, d = params["m"], params["Ixx"], params["Iyy"], params["Izz"], params["CT"], params[
            "CD"], params["d"]

        # Derivatives of vz and vphi, vtheta, vpsi depend on control inputs
        g[:, CrazyFlies.W, CrazyFlies.F_1] = 1 / m
        g[:, CrazyFlies.W, CrazyFlies.F_2] = 1 / m
        g[:, CrazyFlies.W, CrazyFlies.F_3] = 1 / m
        g[:, CrazyFlies.W, CrazyFlies.F_4] = 1 / m

        g[:, CrazyFlies.R, CrazyFlies.F_1] = (1 / Izz) * (-np.sqrt(2) * d)
        g[:, CrazyFlies.R, CrazyFlies.F_2] = (1 / Izz) * (-np.sqrt(2) * d)
        g[:, CrazyFlies.R, CrazyFlies.F_3] = (1 / Izz) * (np.sqrt(2) * d)
        g[:, CrazyFlies.R, CrazyFlies.F_4] = (1 / Izz) * (np.sqrt(2) * d)

        g[:, CrazyFlies.Q, CrazyFlies.F_1] = (1 / Iyy) * (-np.sqrt(2) * d)
        g[:, CrazyFlies.Q, CrazyFlies.F_2] = (1 / Iyy) * (np.sqrt(2) * d)
        g[:, CrazyFlies.Q, CrazyFlies.F_3] = (1 / Iyy) * (np.sqrt(2) * d)
        g[:, CrazyFlies.Q, CrazyFlies.F_4] = (1 / Iyy) * (-np.sqrt(2) * d)

        g[:, CrazyFlies.P, CrazyFlies.F_1] = (1 / Ixx) * (-CD / CT)
        g[:, CrazyFlies.P, CrazyFlies.F_2] = (1 / Ixx) * (CD / CT)
        g[:, CrazyFlies.P, CrazyFlies.F_3] = (1 / Ixx) * (-CD / CT)
        g[:, CrazyFlies.P, CrazyFlies.F_4] = (1 / Ixx) * (CD / CT)

        return g

    # @property
    def u_eq(self):
        u_eq = torch.zeros((1, self.n_controls))
        """ Return the equilibrium state.
            We want total thrust equals m*g.
        """
        u_eq[0, CrazyFlies.F_1] = self.nominal_params["m"] * 9.81 / 4
        u_eq[0, CrazyFlies.F_2] = self.nominal_params["m"] * 9.81 / 4
        u_eq[0, CrazyFlies.F_3] = self.nominal_params["m"] * 9.81 / 4
        u_eq[0, CrazyFlies.F_4] = self.nominal_params["m"] * 9.81 / 4

        return u_eq

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
        K = self.K.type_as(x).to(x.device)
        if op_point is None:
            op_point = self.goal.type_as(x).to(x.device)

        goal = op_point.squeeze().type_as(x).to(x.device)
        u_nominal = -(K @ (x - goal).T).T

        # Adjust for the equilibrium setpoint
        u = u_nominal + self.u_eq().type_as(x).to(x.device)

        # Clamp given the control limits
        upper_u_lim, lower_u_lim = self.control_limits()
        upper_u_lim = upper_u_lim.type_as(x).to(x.device)
        lower_u_lim = lower_u_lim.type_as(x).to(x.device)
        for dim_idx in range(self.n_controls):
            u[:, dim_idx] = torch.clamp(
                u[:, dim_idx],
                min=lower_u_lim[dim_idx].item(),
                max=upper_u_lim[dim_idx].item(),
            )

        return u

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