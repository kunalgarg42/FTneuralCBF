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

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.ones(self.n_dims)
        upper_limit[CrazyFlies.X] = 10.0
        upper_limit[CrazyFlies.Y] = 10.0
        upper_limit[CrazyFlies.Z] = 10.0
        upper_limit[CrazyFlies.U] = 4.0
        upper_limit[CrazyFlies.V] = 4.0
        upper_limit[CrazyFlies.W] = 4.0
        upper_limit[CrazyFlies.PSI] = np.pi / 2.0
        upper_limit[CrazyFlies.THETA] = np.pi / 2.0
        upper_limit[CrazyFlies.PHI] = np.pi / 2.0
        upper_limit[CrazyFlies.R] = np.pi / 4.0
        upper_limit[CrazyFlies.Q] = np.pi / 4.0
        upper_limit[CrazyFlies.P] = np.pi / 4.0

        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control
        limits for this system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.tensor([2, 2, 2, 2])
        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)

    def safe_mask(self, x):
        """Return the mask of x indicating safe regions for the task

        We don't want to crash to the floor, so we need a safe height to avoid the floor

        args:
            x: a tensor of points in the state space
        """
        safe_mask = torch.ones_like(x[:, 0], dtype=torch.bool)

        safe_z = 1.5
        safe_mask = x[:, CrazyFlies.Z] >= safe_z

        return safe_mask

    def unsafe_mask(self, x):
        """Return the mask of x indicating unsafe regions for the task

        args:
            x: a tensor of points in the state space
        """
        unsafe_mask = torch.zeros_like(x[:, 0], dtype=torch.bool)

        unsafe_z = 1.0
        unsafe_mask = x[:, CrazyFlies.Z] <= unsafe_z

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
        psi = x[:, CrazyFlies.PSI]
        theta = x[:, CrazyFlies.THETA]
        phi = x[:, CrazyFlies.PHI]

        u = x[:, CrazyFlies.U]
        v = x[:, CrazyFlies.V]
        w = x[:, CrazyFlies.W]

        r = x[:, CrazyFlies.R]
        q = x[:, CrazyFlies.Q]
        p = x[:, CrazyFlies.P]

        # Derivatives of positions
        f[:, CrazyFlies.X] = w * ( torch.sin(psi) * torch.sin(phi) + torch.cos(psi) * torch.cos(phi) * torch.sin(theta) )\
                             - v * ( torch.sin(psi) * torch.cos(phi) - torch.cos(psi) * torch.sin(phi) * torch.sin(theta) )\
                             + u * torch.cos(psi) * torch.cos(theta)
        f[:, CrazyFlies.Y] = v * ( torch.cos(psi) * torch.cos(phi) + torch.sin(psi) * torch.sin(phi) * torch.sin(theta) )\
                             - w * ( torch.cos(psi) * torch.sin(phi) - torch.sin(psi) * torch.cos(phi) * torch.sin(theta) )\
                             + u * torch.sin(psi) * torch.cos(theta)
        f[:, CrazyFlies.Z] = w * torch.cos(psi) * torch.cos(phi) - u * torch.sin(theta) + v * torch.sin(phi) * torch.cos(theta)

        # Derivatives of angles
        f[:, CrazyFlies.PSI] = p + r * torch.cos(phi) * torch.tan(theta) + q * torch.sin(phi) * torch.tan(theta)
        f[:, CrazyFlies.THETA] = q * torch.cos(phi) - r * torch.sin(phi)
        f[:, CrazyFlies.PHI] = r * torch.cos(phi) / torch.cos(theta) + q * torch.sin(phi) / torch.cos(theta)

        # Derivatives of linear velocities
        f[:, CrazyFlies.U] = r * v - q * w + 9.81 * torch.sin(theta)
        f[:, CrazyFlies.V] = p * w - r * u - 9.81 * torch.sin(phi) * torch.cos(theta)
        f[:, CrazyFlies.W] = q * u - p * v - 9.81 * torch.cos(phi) * torch.cos(theta)

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

    @property
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
