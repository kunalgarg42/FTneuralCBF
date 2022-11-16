"""Define a dynamical system for a Crazyflie"""
from typing import Tuple, List, Optional

import torch
import numpy as np

from .control_affine_system_new import ControlAffineSystemNew


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
        dt: float = 0.01,
        controller_dt: Optional[float] = None,
    ):
        self.x = x
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
        super().__init__(x, nominal_params, dt, controller_dt)

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
        upper_limit[CrazyFlies.Z] = 10.0
        upper_limit[CrazyFlies.U] = 10.0
        upper_limit[CrazyFlies.V] = 10.0
        upper_limit[CrazyFlies.W] = 10.0
        upper_limit[CrazyFlies.PSI] = np.pi / 2.0
        upper_limit[CrazyFlies.THETA] = np.pi / 2.0
        upper_limit[CrazyFlies.PHI] = np.pi / 2.0
        upper_limit[CrazyFlies.R] = np.pi / 2.0
        upper_limit[CrazyFlies.Q] = np.pi / 2.0
        upper_limit[CrazyFlies.P] = np.pi / 2.0

        lower_limit = -1.0 * upper_limit
        lower_limit[CrazyFlies.Z] = 0.01
        # lower_limit = torch.tensor(lower_limit)
        # upper_limit = torch.tensor(upper_limit)

        return (upper_limit, lower_limit)

    def safe_limits(self, sm, sl) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        # define upper and lower limits based around the nominal equilibrium input

        params = self.params
        fault = params["fault"]

        if fault == 0:
            safe_z_l = 1
            safe_z_u = 8
            safe_w_u = 5
            safe_w_l = -5
        else:
            safe_z_l = 0.4
            safe_z_u = 10
            safe_w_u = 6
            safe_w_l = -6

        upper_limit = 0.9 * sm
        lower_limit = 0.9 * sl
        upper_limit[CrazyFlies.Z] = safe_z_u
        lower_limit[CrazyFlies.Z] = safe_z_l
        upper_limit[CrazyFlies.W] = safe_w_u
        lower_limit[CrazyFlies.W] = safe_w_l

        return (upper_limit, lower_limit)

    # @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control
        limits for this system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.tensor([1, 1, 1, 1]) * 0.15
        lower_limit = 0.1 * upper_limit

        # lower_limit = torch.tensor(lower_limit)
        # upper_limit = torch.tensor(upper_limit)

        return (upper_limit, lower_limit)

    def safe_mask(self, x):
        """Return the mask of x indicating safe regions for the task

        We don't want to crash to the floor, so we need a safe height to avoid the floor

        args:
            x: a tensor of points in the state space
        """
        params = self.params
        fault = params["fault"]

        if fault == 0:
            safe_z_l = 1
            safe_z_u = 8
            safe_w_u = 5
            safe_w_l = -5
        else:
            safe_z_l = 1.0
            safe_z_u = 10
            safe_w_u = 2
            safe_w_l = -2

        safe_mask = torch.logical_and(x[:, CrazyFlies.Z] >= safe_z_l, x[:,CrazyFlies.Z] <= safe_z_u)
        # safe_mask.logical_and_(x[:,CrazyFlies.PHI] <= safe_angle)
        # safe_mask.logical_and_(x[:,CrazyFlies.PHI] >= -safe_angle)
        # safe_mask.logical_and_(x[:,CrazyFlies.THETA] <= safe_angle)
        # safe_mask.logical_and_(x[:,CrazyFlies.THETA] >= -safe_angle)
        safe_mask.logical_and_(x[:, CrazyFlies.W] >= safe_w_l)
        safe_mask.logical_and_(x[:, CrazyFlies.W] <= safe_w_u)

        # x.norm(dim=-1) >= unsafe_radius

        return safe_mask

    def unsafe_mask(self, x):
        """Return the mask of x indicating unsafe regions for the task

        args:
            x: a tensor of points in the state space
        """
        params = self.params
        fault = params["fault"]

        unsafe_mask = torch.zeros_like(x[:, 0], dtype=torch.bool)

        if fault == 0:
            unsafe_z_l = 0.5
            unsafe_z_u = 9
            unsafe_w_l = -5.5
            unsafe_w_u = 5.5
        else:
            unsafe_z_l = 0.5
            unsafe_z_u = 12
            unsafe_w_l = -2.5
            unsafe_w_u = 2.5

        unsafe_mask = torch.logical_or(x[:, CrazyFlies.Z] <= unsafe_z_l, x[:,CrazyFlies.Z] >= unsafe_z_u)
        # unsafe_mask.logical_or_(x[:,CrazyFlies.PHI] >= unsafe_angle)
        # unsafe_mask.logical_or_(x[:,CrazyFlies.PHI] <= - unsafe_angle)
        # unsafe_mask.logical_or_(x[:,CrazyFlies.THETA] >= unsafe_angle)
        # unsafe_mask.logical_or_(x[:,CrazyFlies.THETA] >= - unsafe_angle)
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
        f = torch.zeros((batch_size, self.n_dims, 1))
        f = f.type_as(x)

        # Extract the needed parameters
        m, Ixx, Iyy, Izz, CT, CD, d = params["m"], params["Ixx"], params["Iyy"], params["Izz"], params["CT"], params["CD"], params["d"]

        # Extract state variables
        psi = x[:, CrazyFlies.PSI].reshape(batch_size,1)
        theta = x[:, CrazyFlies.THETA].reshape(batch_size,1)
        phi = x[:, CrazyFlies.PHI].reshape(batch_size,1)

        u = x[:, CrazyFlies.U].reshape(batch_size,1)
        v = x[:, CrazyFlies.V].reshape(batch_size,1)
        w = x[:, CrazyFlies.W].reshape(batch_size,1)

        r = x[:, CrazyFlies.R].reshape(batch_size,1)
        q = x[:, CrazyFlies.Q].reshape(batch_size,1)
        p = x[:, CrazyFlies.P].reshape(batch_size,1)

        s_psi = torch.sin(psi).reshape(batch_size,1)
        s_phi = torch.sin(phi).reshape(batch_size,1)
        s_the = torch.sin(theta).reshape(batch_size,1)

        c_psi = torch.cos(psi).reshape(batch_size,1)
        c_phi = torch.cos(phi).reshape(batch_size,1)
        c_the = torch.cos(theta).reshape(batch_size,1)

        t_psi = torch.tan(psi).reshape(batch_size,1)
        t_phi = torch.tan(phi).reshape(batch_size,1)
        t_the = torch.tan(theta).reshape(batch_size,1)

        # Derivatives of positions
        f[:, CrazyFlies.X] = w * ( s_psi * s_phi + c_psi * c_phi * s_the )\
                             - v * ( s_psi * c_phi - c_psi * s_phi * s_the )\
                             + u * c_psi * c_the
        f[:, CrazyFlies.Y] = v * ( c_psi * c_phi + s_psi * s_phi * s_the )\
                             - w * ( c_psi * s_phi - s_psi * c_phi * s_the )\
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
        f[:, CrazyFlies.R] = ( Ixx - Iyy ) / Izz * p * q
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
        g = torch.zeros((batch_size, self.n_dims, self.n_controls))
        g = g.type_as(x)

        # Extract the needed parameters
        m, Ixx, Iyy, Izz, CT, CD, d = params["m"], params["Ixx"], params["Iyy"], params["Izz"], params["CT"], params[
            "CD"], params["d"]

        # Derivatives of vz and vphi, vtheta, vpsi depend on control inputs
        g[:, CrazyFlies.W, CrazyFlies.F_1] = 1 / m
        g[:, CrazyFlies.W, CrazyFlies.F_2] = 1 / m
        g[:, CrazyFlies.W, CrazyFlies.F_3] = 1 / m
        g[:, CrazyFlies.W, CrazyFlies.F_4] = 1 / m

        g[:, CrazyFlies.R, CrazyFlies.F_1] = ( 1 / Izz ) * ( -np.sqrt(2) * d )
        g[:, CrazyFlies.R, CrazyFlies.F_2] = ( 1 / Izz ) * ( -np.sqrt(2) * d )
        g[:, CrazyFlies.R, CrazyFlies.F_3] = ( 1 / Izz ) * ( np.sqrt(2) * d )
        g[:, CrazyFlies.R, CrazyFlies.F_4] = ( 1 / Izz ) * ( np.sqrt(2) * d )

        g[:, CrazyFlies.Q, CrazyFlies.F_1] = ( 1 / Iyy )* ( -np.sqrt(2) * d )
        g[:, CrazyFlies.Q, CrazyFlies.F_2] = ( 1 / Iyy ) * ( np.sqrt(2) * d )
        g[:, CrazyFlies.Q, CrazyFlies.F_3] = ( 1 / Iyy ) * ( np.sqrt(2) * d )
        g[:, CrazyFlies.Q, CrazyFlies.F_4] = ( 1 / Iyy ) * (-np.sqrt(2) * d )

        g[:, CrazyFlies.P, CrazyFlies.F_1] = ( 1 / Ixx ) * ( -CD / CT )
        g[:, CrazyFlies.P, CrazyFlies.F_2] = ( 1 / Ixx ) * ( CD / CT )
        g[:, CrazyFlies.P, CrazyFlies.F_3] = ( 1 / Ixx ) * ( -CD / CT )
        g[:, CrazyFlies.P, CrazyFlies.F_4] = ( 1 / Ixx ) * ( CD / CT )

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
