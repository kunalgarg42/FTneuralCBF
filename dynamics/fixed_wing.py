"""Define a dynamical system for a 3D quadrotor"""
from typing import Tuple, List, Optional

import torch
import numpy as np

from .control_affine_system_new import ControlAffineSystemNew
# from .utils import grav, Scenario, ScenarioList
# from neural_clbf.systems.utils import grav, Scenario, ScenarioList


class FixedWing(ControlAffineSystemNew):
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

    # Number of states and controls
    N_DIMS = 9
    N_CONTROLS = 4

    # State indices
    
    V = 0
    ALPHA = 1
    BETA = 2

    PHI = 3
    GAMMA = 4
    PSI = 5

    P = 6
    Q = 7
    R = 8


    # Control indices
    T = 0
    DA = 1
    DE = 2
    DR = 3

    # self.x = x
    # self.params = params



    def __init__(
        self,
        x: torch.Tensor,
        nominal_params,
        dt: float = 0.01,
        controller_dt: Optional[float] = None,
        # scenarios: Optional[ScenarioList] = None,
        ):
        self.x = x
        self.params = nominal_params
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
        super().__init__(x, nominal_params, dt, controller_dt)

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
        valid = valid and "m" in params
        valid = valid and "g" in params
        valid = valid and "Ixx" in params
        valid = valid and "Iyy" in params
        valid = valid and "Izz" in params
        valid = valid and "Ixz" in params
        valid = valid and "S" in params
        valid = valid and "b" in params
        valid = valid and "bar_c" in params
        valid = valid and "rho" in params
        valid = valid and "Cd0" in params
        valid = valid and "Cda" in params
        valid = valid and "Clb" in params
        valid = valid and "Clda" in params
        valid = valid and "Cldr" in params
        valid = valid and "Clp" in params
        valid = valid and "Clr" in params
        valid = valid and "Cm0" in params
        valid = valid and "Cmde" in params
        valid = valid and "Cma" in params
        valid = valid and "Cmq" in params
        valid = valid and "Cnb" in params
        valid = valid and "Cnda" in params
        valid = valid and "Cndr" in params
        valid = valid and "Cnr" in params
        valid = valid and "Cndr" in params
        valid = valid and "Cyb" in params
        valid = valid and "Cyp" in params
        valid = valid and "Cyr" in params
        valid = valid and "Cydr" in params
        valid = valid and "Cz0" in params
        valid = valid and "Cza" in params
        valid = valid and "Czq" in params
        valid = valid and "Czde" in params
        valid = valid and "Cx0" in params
        valid = valid and "Cxq" in params



        # Make sure all parameters are physically valid
        valid = valid and params["m"] > 0

        return valid

    @property
    def n_dims(self) -> int:
        return FixedWing.N_DIMS

    @property
    def angle_dims(self) -> List[int]:
        return [FixedWing.PHI, FixedWing.GAMMA, FixedWing.PSI]

    @property
    def n_controls(self) -> int:
        return FixedWing.N_CONTROLS

    # @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.ones(self.n_dims)
        upper_limit[FixedWing.V] = 200.0
        upper_limit[FixedWing.ALPHA] = np.pi / 3.0
        upper_limit[FixedWing.BETA] = np.pi / 6.0
        upper_limit[FixedWing.PHI] = np.pi / 2.0
        upper_limit[FixedWing.GAMMA] = np.pi / 2.0
        upper_limit[FixedWing.PSI] = np.pi / 2.0
        upper_limit[FixedWing.P] = 4
        upper_limit[FixedWing.Q] = 4
        upper_limit[FixedWing.R] = 4

        lower_limit = -1.0 * upper_limit
        lower_limit[FixedWing.V] = 0.0

        lower_limit = torch.tensor(lower_limit)
        upper_limit = torch.tensor(upper_limit)

        return (upper_limit, lower_limit)

    # @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control
        limits for this system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.tensor([250, 1, 1, 1])
        lower_limit = -1.0 * upper_limit
        lower_limit[FixedWing.T] = 0

        lower_limit = torch.tensor(lower_limit)
        upper_limit = torch.tensor(upper_limit)

        return (upper_limit, lower_limit)

    def safe_mask(self, x):
        """Return the mask of x indicating safe regions for the obstacle task

        args:
            x: a tensor of points in the state space
        """
        safe_mask = torch.ones_like(x[:, 0], dtype=torch.bool)

        safe_alpha = np.pi / 6.0
        safe_alpha_l = np.pi / 60.0
        safe_beta = np.pi / 12
        # safe_radius = 3
        safe_mask = torch.logical_and(
            x[:, FixedWing.ALPHA] <= safe_alpha, x[:, FixedWing.ALPHA] >= safe_alpha_l)
        safe_mask = torch.logical_and(safe_mask, x[:, FixedWing.BETA] <= safe_beta)
        safe_mask = torch.logical_and(safe_mask, x[:, FixedWing.BETA] >= -safe_beta #, x.norm(dim=-1) <= safe_radius
        )

        return safe_mask

    def unsafe_mask(self, x):
        """Return the mask of x indicating unsafe regions for the obstacle task

        args:
            x: a tensor of points in the state space
        """
        unsafe_mask = torch.zeros_like(x[:, 0], dtype=torch.bool)

        # We have a floor that we need to avoid and a radius we need to stay inside of
        unsafe_alpha = np.pi / 5.5
        unsafe_alpha_l = np.pi / 50.0
        unsafe_beta = np.pi / 11

        unsafe_mask = torch.logical_or(
            x[:, FixedWing.ALPHA] >= unsafe_alpha, x[:, FixedWing.ALPHA] <= unsafe_alpha_l)
        unsafe_mask = torch.logical_or(unsafe_mask, x[:, FixedWing.BETA] >= unsafe_beta)
        unsafe_mask = torch.logical_or(unsafe_mask, x[:, FixedWing.BETA] <= -unsafe_beta
        )

        return unsafe_mask

    def goal_mask(self, x):
        """Return the mask of x indicating points in the goal set (within 0.2 m of the
        goal).

        args:
            x: a tensor of points in the state space
        """
        goal_mask = torch.ones_like(x[:, 0], dtype=torch.bool)

        # Define the goal region as being near the goal
        near_goal = torch.logical_and(x[:,FixedWing.V]<= 52, x[:,FixedWing.V] >= 48)
        # near_goal = x.norm(dim=-1) <= 0.3
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
        
        m = params["m"]
        grav = params["g"]
        Ixx = params["Ixx"]
        Iyy = params["Iyy"]
        Izz = params["Izz"]
        Ixz = params["Ixz"]
        
        S = params["S"]
        b = params["b"]
        bar_c = params["bar_c"]
        rho = params["rho"]
        
        Cd0 = params["Cd0"]
        Cda = params["Cda"]

        Clb = params["Clb"]
        Clda = params["Clda"]
        Cldr = params["Cldr"]
        Clp = params["Clp"]
        Clr = params["Clr"]

        Cm0 = params["Cm0"]
        Cma = params["Cma"]
        Cmde = params["Cmde"]
        Cmq = params["Cmq"]

        Cnb = params["Cnb"]
        Cndr = params["Cndr"]
        Cnda = params["Cnda"]
        Cnp = params["Cnp"]
        Cnr = params["Cnr"]

        Cyb = params["Cyb"]
        Cydr = params["Cydr"]
        Cyp = params["Cyp"]
        Cyr = params["Cyr"]

        Cz0 = params["Cz0"]
        Cza = params["Cza"]
        Czde = params["Czde"]
        Czq = params["Czq"]  

        Cx0 = params["Cx0"]
        Cxq = params["Cxq"]      




        I_Gamma = Ixx * Izz - Ixz * Ixz

        c1 = (Ixx - Iyy + Izz) * Ixz / I_Gamma
        c2 = (Izz * Izz - Iyy * Izz + Ixz * Ixz) / I_Gamma
        c3 = Izz / I_Gamma
        c4 = Ixz / I_Gamma
        c5 = (Izz - Ixx) / Iyy
        c6 = Ixz / Iyy
        c7 = 1.0 / Iyy
        c8 = (Ixx * Ixx - Ixx * Iyy + Ixz * Ixz) / I_Gamma
        c9 = Ixx / I_Gamma



        # print(FixedWing.V)

        V = x[:, FixedWing.V].reshape(batch_size,1)

        # print(V)

        alpha = x[:,FixedWing.ALPHA].reshape(batch_size,1)
        beta = x[:,FixedWing.BETA].reshape(batch_size,1)
        phi = x[:,FixedWing.PHI].reshape(batch_size,1)
        gamma = x[:,FixedWing.GAMMA].reshape(batch_size,1)
        psi = x[:,FixedWing.PSI].reshape(batch_size,1)
        p = x[:,FixedWing.P].reshape(batch_size,1)
        q = x[:,FixedWing.Q].reshape(batch_size,1)
        r = x[:,FixedWing.R].reshape(batch_size,1)



        
        s_a = torch.sin(alpha).reshape(batch_size,1)
        c_a = torch.cos(alpha).reshape(batch_size,1)

        s_b = torch.sin(beta).reshape(batch_size,1)
        c_b = torch.cos(beta).reshape(batch_size,1)
        t_b = torch.tan(beta).reshape(batch_size,1)

        s_g = torch.sin(gamma).reshape(batch_size,1)
        c_g = torch.cos(gamma).reshape(batch_size,1)
        t_g = torch.tan(gamma).reshape(batch_size,1)

        s_p = torch.sin(phi).reshape(batch_size,1)
        c_p = torch.cos(phi).reshape(batch_size,1)



        bar_p = 1.0 / 2.0 * rho * rho * V * V * S

        # D = Cd0 * bar_p

        FX = (Cx0 + Cxq * (q * bar_c / 2 / V)) * bar_p

        FY = - (Cyb * beta + b / 2 / V * (Cyp * p + Cyr * r)) * bar_p

        FZ = (Cz0 + Cza * alpha + q * bar_c / 2 / V * Czq) * bar_p

        # print("366")

        # print(V)

        # print(FX)

        Fx_t = FX.reshape(batch_size,1)
        # Fx_t = Fx_t.detach().numpy()
        # Fx_t = np.array(Fx_t).reshape(batch_size,1)
        Fy_t = FY.reshape(batch_size,1)
        # Fy_t = Fy_t.detach().numpy()
        # Fy_t = np.array(Fy_t).reshape(batch_size,1)
        Fz_t = FZ.reshape(batch_size,1)
        # Fz_t = Fz_t.detach().numpy()
        # Fz_t = np.array(Fz_t).reshape(batch_size,1)
        # Fbf = torch.tensor([FX, FY, FZ])
        # Fbf = np.array([[Fx_t], [Fy_t], [Fz_t]]).reshape(3,batch_size)
        # Fbf = Fbf.detach().numpy()
        # Fbf = np.array(Fbf).reshape(3,1)

        Rw11 = c_a * c_b 
        # Rw11 = Rw11.detach().numpy()
        Rw11 = Rw11.reshape(batch_size,1)
        
        Rw12 = s_b 
        # Rw12 = Rw12.detach().numpy()
        Rw12 = Rw12.reshape(batch_size,1)
        
        Rw13 =  s_a * c_b
        # Rw13 = Rw13.detach().numpy()
        Rw13 = Rw13.reshape(batch_size,1)
        
        Rw21 = -c_a * s_b
        # Rw21 = Rw21.detach().numpy()
        Rw21 = Rw21.reshape(batch_size,1)
        
        Rw22 = c_b 
        # Rw22 = Rw22.detach().numpy()
        Rw22 = Rw22.reshape(batch_size,1)
        
        Rw23 = -s_a * s_b
        # Rw23 = Rw23.detach().numpy()
        Rw23 = Rw23.reshape(batch_size,1)
        
        Rw31 = -s_a 
        # Rw31 = Rw31.detach().numpy()
        Rw31 = Rw31.reshape(batch_size,1)
        
        Rw32 = torch.tensor([0.0]*batch_size).reshape(batch_size,1)
        # Rw32 = Rw32.detach().numpy()
        # Rw32 = np.array(Rw32).reshape(batch_size,1)
        
        Rw33 = c_a
        # Rw33 = Rw33.detach().numpy()
        Rw33 = Rw33.reshape(batch_size,1)

        # Rw1 = np.array([Rw11, Rw12, Rw13]).reshape(batch_size,3)
        # Rw2 = np.array([Rw21, Rw22, Rw23]).reshape(batch_size,3)
        # Rw3 = np.array([Rw31, Rw32, Rw33]).reshape(batch_size,3)
        # Rwb = np.array([Rw1, Rw2, Rw3]).reshape(3,batch_size,3)
        # Rwb = torch.tensor([, , ])
        # Rwb = Rwb.detach().numpy()
        # Rwb = np.array(Rwb).reshape(3,3)

        # Fwf = np.matmul(Rwb, Fbf) # .reshape(3, batch_size)


        # print(Rwb)
        # Fwf = np.array(Fwf, dtype = float).reshape(3,batch_size)


        # print(Fwf)
        # if batch_size == 64:
        #     print(np.multiply(Rw11, Fx_t))

        #     print(Fwf)

        D = torch.multiply(Rw11, Fx_t) + torch.multiply(Rw12, Fy_t) + torch.multiply(Rw13, Fz_t)

        Y = torch.multiply(Rw21, Fx_t) + torch.multiply(Rw32, Fy_t) + torch.multiply(Rw23, Fz_t)

        L = torch.multiply(Rw31, Fx_t) + torch.multiply(Rw32, Fy_t) + torch.multiply(Rw33, Fz_t)

        Fwx = - D
        # Fwx = torch.tensor([- D]).reshape(batch_size,1)

        # print(batch_size)

        # print(Fwf)

        # Fwy = torch.tensor([- Y]).reshape(batch_size,1)
        # Fwz = torch.tensor([- L]).reshape(batch_size,1)
        Fwy = - Y 
        Fwz = - L

        # if batch_size == 64:
        #     # print(Fwx)
        #     print(FZ.shape)
        #     print(s_g.shape)

        # L = (Clb * beta + b / 2 / V * (Clp * p + Clr * r)) * bar_p

        La = b * (Clb * beta + b / 2 / V * (Clp * p + Clr * r)) * bar_p

        Ma = bar_c * (Cm0 + Cma * alpha + bar_c / 2 / V *(Cmq * q)) * bar_p

        Na = b * (Cnb * beta + b / 2 / V *(Cnp * p + Cnr * r)) * bar_p

        # print(Fwf)

        
        f = torch.zeros((batch_size, self.n_dims, 1))
        f = f.type_as(x)


        f[:, FixedWing.V] = - grav * s_g + Fwx / m

        # f[:, FixedWing.V] = torch.matmul(-1.0*grav,torch.sin(gamma))
        
        # qwf = -1.0* torch.div(torch.matmul(grav,torch.matmul(torch.cos(gamma),torch.cos(phi))),V)

        qwf = -1.0 * grav * c_g * c_p / V - 1.0 * Fwz / m / V

        # if batch_size == 64:
        #     print(V.shape)
        #     print((-1.0 * torch.multiply(c_p,c_g)).shape)
        #     print(torch.div(Fwz,V).shape)            

        # f[:, FixedWing.ALPHA] = q - torch.matmul((torch.matmul(p, torch.cos(alpha))+torch.matmul(r,torch.sin(alpha))),torch.tan(beta)) -torch.div(qwf,torch.cos(beta))
        f[:, FixedWing.ALPHA] = q - qwf / c_b - (p * c_a + r * s_a) * t_b
        
        # rwf = torch.div(torch.matmul(grav,torch.matmul(torch.cos(gamma),torch.sin(phi))),V) 
        rwf = grav * c_g * s_p / V + Fwy / m / V

        # f[:, FixedWing.BETA] = rwf + torch.matmul(p,torch.sin(alpha))-torch.matmul(r,torch.cos(alpha))
        f[:, FixedWing.BETA] = rwf + p * s_a - r * c_a

        # pw = torch.matmul(p,torch.matmul(torch.cos(alpha),torch.cos(beta))) + torch.matmul(q-f[:, FixedWing.ALPHA],torch.sin(beta)) + torch.matmul(r,torch.matmul(torch.sin(alpha),torch.cos(beta)))
        pw = p * c_a * c_b + (q-f[:,FixedWing.ALPHA]) * s_b + r * s_a * c_b
        
        # f[:,FixedWing.PHI] = pw + torch.matmul(torch.matmul(qwf,sin(phi))+torch.matmul(rwf,torch.cos(phi)),torch.tan(gamma))
        f[:,FixedWing.PHI] = pw + (qwf * s_p + rwf * c_p) * t_g

        f[:,FixedWing.GAMMA] = qwf * c_g - rwf * s_g
        
        f[:,FixedWing.PSI] = (qwf * s_p + rwf * c_p) / c_g

        f[:,FixedWing.P] = c1 * p * q - c2 * q * r +  c3 * La + c4 * Na

        f[:, FixedWing.Q] = c5 * p * r - c6* (p * p - r * r) + c7 * Ma

        f[:, FixedWing.R] = c8 * p * q - c1 * q * r + c4 * La + c9 * Na

        
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
        # print(x.shape)
        # x = torch.tensor(x)
        batch_size = x.shape[0]
        g = torch.zeros((batch_size, self.n_dims, self.n_controls))
        g = g.type_as(x)

        # Extract the needed parameters
        m = params["m"]
        grav = params["g"]
        Ixx = params["Ixx"]
        Iyy = params["Iyy"]
        Izz = params["Izz"]
        Ixz = params["Ixz"]
        
        S = params["S"]
        b = params["b"]
        bar_c = params["bar_c"]
        rho = params["rho"]
        
        Cd0 = params["Cd0"]
        Cda = params["Cda"]

        Clb = params["Clb"]
        Clda = params["Clda"]
        Cldr = params["Cldr"]
        Clp = params["Clp"]
        Clr = params["Clr"]

        Cm0 = params["Cm0"]
        Cma = params["Cma"]
        Cmde = params["Cmde"]
        Cmq = params["Cmq"]

        Cnb = params["Cnb"]
        Cndr = params["Cndr"]
        Cnda = params["Cnda"]
        Cnp = params["Cnp"]
        Cnr = params["Cnr"]

        Cyb = params["Cyb"]
        Cydr = params["Cydr"]
        Cyp = params["Cyp"]
        Cyr = params["Cyr"]

        Cz0 = params["Cz0"]
        Cza = params["Cza"]
        Czde = params["Czde"]
        Czq = params["Czq"]  

        Cx0 = params["Cx0"]
        Cxq = params["Cxq"]



        I_Gamma = Ixx * Izz - Ixz * Ixz

        c1 = (Ixx - Iyy + Izz) * Ixz / I_Gamma
        c2 = (Izz * Izz - Iyy * Izz + Ixz * Ixz) / I_Gamma
        c3 = Izz / I_Gamma
        c4 = Ixz / I_Gamma
        c5 = (Izz - Ixx) / Iyy
        c6 = Ixz / Iyy
        c7 = 1.0 / Iyy
        c8 = (Ixx * Ixx - Ixx * Iyy + Ixz * Ixz) / I_Gamma
        c9 = Ixx / I_Gamma

        V = x[:, FixedWing.V].reshape(batch_size)
        alpha = x[:,FixedWing.ALPHA].reshape(batch_size)
        beta = x[:,FixedWing.BETA].reshape(batch_size)
        phi = x[:,FixedWing.PHI].reshape(batch_size)
        gamma = x[:,FixedWing.GAMMA].reshape(batch_size)
        psi = x[:,FixedWing.PSI].reshape(batch_size)
        p = x[:,FixedWing.P].reshape(batch_size)
        q = x[:,FixedWing.Q].reshape(batch_size)
        r = x[:,FixedWing.R].reshape(batch_size)



        
        s_a = torch.sin(alpha).reshape(batch_size)
        c_a = torch.cos(alpha).reshape(batch_size)

        s_b = torch.sin(beta).reshape(batch_size)
        c_b = torch.cos(beta).reshape(batch_size)
        t_b = torch.tan(beta).reshape(batch_size)

        s_g = torch.sin(gamma).reshape(batch_size)
        c_g = torch.cos(gamma).reshape(batch_size)
        t_g = torch.tan(gamma).reshape(batch_size)

        s_p = torch.sin(phi).reshape(batch_size)
        c_p = torch.cos(phi).reshape(batch_size)



        bar_p = 1.0 / 2.0 * rho * rho * V * V * S

        Rw11 = c_a * c_b 
        # Rw11 = Rw11.detach().numpy()
        Rw11 = Rw11.reshape(batch_size)
        
        Rw12 = s_b 
        # Rw12 = Rw12.detach().numpy()
        Rw12 = Rw12.reshape(batch_size)
        
        Rw13 =  s_a * c_b
        # Rw13 = Rw13.detach().numpy()
        Rw13 = Rw13.reshape(batch_size)
        
        Rw21 = -c_a * s_b
        # Rw21 = Rw21.detach().numpy()
        Rw21 = Rw21.reshape(batch_size)
        
        Rw22 = c_b 
        # Rw22 = Rw22.detach().numpy()
        Rw22 = Rw22.reshape(batch_size)
        
        Rw23 = -s_a * s_b
        # Rw23 = Rw23.detach().numpy()
        Rw23 = Rw23.reshape(batch_size)
        
        Rw31 = -s_a 
        # Rw31 = Rw31.detach().numpy()
        Rw31 = Rw31.reshape(batch_size)
        
        Rw32 = torch.tensor([0.0]*batch_size).reshape(batch_size)
        # Rw32 = Rw32.detach().numpy()
        # Rw32 = np.array(Rw32).reshape(batch_size,1)
        
        Rw33 = c_a
        # Rw33 = Rw33.detach().numpy()
        Rw33 = Rw33.reshape(batch_size)

        # D = Cd0 * bar_p


        FXda = torch.tensor([0.0]*batch_size).reshape(batch_size)
        FXdr = FXda
        FXde = FXda

        FYda = FXda
        FYdr = - (Cydr) * bar_p
        FYde = FXda

        FZda = FXda
        FZdr = FXda
        FZde = (Czde) * bar_p

        # Fda = torch.tensor([FXda, FYda, FZda])
        # Fda = Fda.detach().numpy()
        # Fbfda = np.array(Fda).reshape(3,1)
        
        # Fdr = torch.tensor([FXdr, FYdr, FZdr])
        # Fdr = Fdr.detach().numpy()
        # Fbfdr = np.array(Fdr).reshape(3,1)
        
        # Fde = torch.tensor([FXde, FYde, FZde])
        # Fde = Fde.detach().numpy()
        # Fbfde = np.array(Fde).reshape(3,1)

        # Rwb = torch.tensor([[c_a * c_b, s_b , s_a * c_b], [-c_a * s_b, c_b , -s_a * s_b], [-s_a, 0.0, c_a]])
        # Rwb = Rwb.detach().numpy()

        # Rwb = np.array(Rwb).reshape(3,3)



        # Fwfda = np.dot(Rwb, Fbfda)
        # Fwfdr = np.dot(Rwb, Fbfdr)
        # Fwfde = np.dot(Rwb, Fbfde)

        Dda = torch.multiply(Rw11, FXda) + torch.multiply(Rw12, FYda) + torch.multiply(Rw13, FZda)
        Dde = torch.multiply(Rw11, FXde) + torch.multiply(Rw12, FYde) + torch.multiply(Rw13, FZde)
        Ddr = torch.multiply(Rw11, FXdr) + torch.multiply(Rw12, FYdr) + torch.multiply(Rw13, FZdr)        

        # Dda = Fwfda[0]
        # Dde = Fwfde[0]
        # Ddr = Fwfdr[0]

        # Yda = Fwfda[1]
        # Yde = Fwfde[1]
        # Ydr = Fwfdr[1]
        Yda = torch.multiply(Rw21, FXda) + torch.multiply(Rw22, FYda) + torch.multiply(Rw23, FZda)
        Yde = torch.multiply(Rw21, FXde) + torch.multiply(Rw22, FYde) + torch.multiply(Rw23, FZde)
        Ydr = torch.multiply(Rw21, FXdr) + torch.multiply(Rw22, FYdr) + torch.multiply(Rw23, FZdr)

        # Lda = Fwfda[2]
        # Lde = Fwfde[2]
        # Ldr = Fwfdr[2]

        Lda = torch.multiply(Rw31, FXda) + torch.multiply(Rw32, FYda) + torch.multiply(Rw33, FZda)
        Lde = torch.multiply(Rw31, FXde) + torch.multiply(Rw32, FYde) + torch.multiply(Rw33, FZde)
        Ldr = torch.multiply(Rw31, FXdr) + torch.multiply(Rw32, FYdr) + torch.multiply(Rw33, FZdr)

        # Fwxda = torch.tensor([- Dda])
        # Fwxdr = torch.tensor([- Ddr])
        # Fwxde = torch.tensor([- Dde])
        # Fwyda = torch.tensor([- Yda])
        # Fwydr = torch.tensor([- Ydr])
        # Fwyde = torch.tensor([- Yde])
        # Fwzda = torch.tensor([- Lda])
        # Fwzde = torch.tensor([- Lde])
        # Fwzdr = torch.tensor([- Ldr])

        Fwxda = - Dda
        Fwxdr = - Ddr
        Fwxde = - Dde
        Fwyda = - Yda
        Fwydr = - Ydr
        Fwyde = - Yde
        Fwzda = - Lda
        Fwzde = - Lde
        Fwzdr = - Ldr



        Lada = b * (Clda) * bar_p
        Ladr = b * (Clr) * bar_p
        Lade = torch.tensor([0.0]*batch_size).reshape(batch_size)

        Made = bar_c * (Cmde) * bar_p
        Mada = Lade
        Madr = Lade

        Nada = b * (Cnda) * bar_p
        Nadr = b * (Cndr) * bar_p
        Nade = Lade

        # if batch_size == 64:
        #     print((c_a*c_b).shape)
        #     print(g[:, FixedWing.V, FixedWing.T].shape)

        g[:, FixedWing.V,FixedWing.T] = (c_a * c_b / m)

        g[:, FixedWing.V,FixedWing.DA] =  Fwxda / m

        g[:, FixedWing.V,FixedWing.DE] =  Fwxde / m

        g[:, FixedWing.V,FixedWing.DR] =  Fwxdr / m



        g[:, FixedWing.ALPHA, FixedWing.T] =  - s_a / m / V / c_b
        g[:, FixedWing.ALPHA, FixedWing.DR] =  Fwzdr/ m / V / c_b
        g[:, FixedWing.ALPHA, FixedWing.DA] =  Fwzda/ m / V / c_b
        g[:, FixedWing.ALPHA, FixedWing.DE] =  Fwzde/ m / V / c_b

        g[:, FixedWing.BETA, FixedWing.T] = -c_a * s_b / m / V
        g[:, FixedWing.BETA, FixedWing.DA] =  Fwyda / m / V
        g[:, FixedWing.BETA, FixedWing.DE] =  Fwyde / m / V
        g[:, FixedWing.BETA, FixedWing.DR] =  Fwydr / m / V

        g[:, FixedWing.PHI,FixedWing.T] = s_p / m / V * s_a * t_g + c_p * t_g / m / V * (-c_a * s_b)
        g[:, FixedWing.PHI, FixedWing.DA] = -s_p / m / V *Fwzda * t_g + c_p * t_g / m / V * Fwyda
        g[:, FixedWing.PHI, FixedWing.DR] = -s_p / m / V *Fwzdr * t_g + c_p * t_g / m / V * Fwyde
        g[:, FixedWing.PHI, FixedWing.DE] = -s_p / m / V *Fwzdr * t_g + c_p * t_g / m / V * Fwyde

        g[:, FixedWing.GAMMA, FixedWing.T] = - c_p / m / V * c_a * c_b  - s_p / m / V * (-c_a * s_b)
        g[:, FixedWing.GAMMA, FixedWing.DA] = -c_p / m / V * Fwzda - s_p / m / V * Fwyda
        g[:, FixedWing.GAMMA, FixedWing.DE] = -c_p / m / V * Fwzde - s_p / m / V * Fwyde
        # print(Fwzdr)
        g[:, FixedWing.GAMMA, FixedWing.DR] = -c_p * Fwzdr / m / V  - s_p * Fwydr / m / V 

        g[:, FixedWing.PSI, FixedWing.T] = - s_p / m / V * c_a * c_b / c_g  + c_p / m / V * (-c_a * s_b) / c_g
        g[:, FixedWing.PSI, FixedWing.DA] = -s_p / m / V * Fwzda / c_g + c_p / m / V * Fwyda / c_g
        g[:, FixedWing.PSI, FixedWing.DE] = -s_p / m / V * Fwzde / c_g + c_p / m / V * Fwyde / c_g
        g[:, FixedWing.PSI, FixedWing.DR] = -s_p / m / V * Fwzdr / c_g + c_p / m / V * Fwydr / c_g

        g[:, FixedWing.P, FixedWing.DA] = c3 * Lada + c4 * Nada
        g[:, FixedWing.P, FixedWing.DE] = c3 * Lade + c4 * Nade
        g[:, FixedWing.P, FixedWing.DR] = c3 * Ladr + c4 * Nadr

        g[:, FixedWing.Q, FixedWing.DA] = c7 * Mada
        g[:, FixedWing.Q, FixedWing.DE] = c7 * Made
        g[:, FixedWing.Q, FixedWing.DR] = c7 * Madr

        g[:, FixedWing.R, FixedWing.DA] = c4 * Lada + c9 * Nada
        g[:, FixedWing.R, FixedWing.DE] = c4 * Lade + c9 * Nade
        g[:, FixedWing.R, FixedWing.DR] = c4 * Ladr + c9 * Nadr

        # print(g)




        # Derivatives of all orientations are control variables
        # g[:, FixedWing.PHI :, FixedWing.PHI_DOT :] = torch.eye(self.n_controls - 1)

        return g

    def u_in(self):
        x = self.x
        params = self.params
        grav = params["g"]
        m = params["m"]

        rho = params["rho"]
        S = params["S"]
        bar_c = params["bar_c"]
        b = params["b"]

        CCd0 = params["Cd0"]
        Cda = params["Cda"]

        Clb = params["Clb"]
        Clda = params["Clda"]
        Cldr = params["Cldr"]
        Clp = params["Clp"]
        Clr = params["Clr"]

        Cm0 = params["Cm0"]
        Cma = params["Cma"]
        Cmde = params["Cmde"]
        Cmq = params["Cmq"]

        Cnb = params["Cnb"]
        Cndr = params["Cndr"]
        Cnda = params["Cnda"]
        Cnp = params["Cnp"]
        Cnr = params["Cnr"]

        Cyb = params["Cyb"]
        Cydr = params["Cydr"]
        Cyp = params["Cyp"]
        Cyr = params["Cyr"]

        Cz0 = params["Cz0"]
        Cza = params["Cza"]
        Czde = params["Czde"]
        Czq = params["Czq"]  

        Cx0 = params["Cx0"]
        Cxq = params["Cxq"]


        # print(x)
        # x = np.array(x)
        # x = torch.from_numpy(x)

        # V = x[:,FixedWing.V]
        V = x[:,FixedWing.V]
        alpha = x[:,FixedWing.ALPHA]
        beta = x[:,FixedWing.BETA]
        phi = x[:,FixedWing.PHI]
        gamma = x[:,FixedWing.GAMMA]
        psi = x[:,FixedWing.PSI]
        p = x[:,FixedWing.P]
        q = x[:,FixedWing.Q]
        r = x[:,FixedWing.R]

        bar_p = 1.0 / 2.0 * rho * rho * V * V * S



        
        s_a = torch.sin(alpha)
        c_a = torch.cos(alpha)

        s_b = torch.sin(beta)
        c_b = torch.cos(beta)
        t_b = torch.tan(beta)

        s_g = torch.sin(gamma)
        c_g = torch.cos(gamma)
        t_g = torch.tan(gamma)

        s_p = torch.sin(phi)
        c_p = torch.cos(phi)

        u_eq = torch.zeros((1, self.n_controls))
        # print(u_eq)

        u_eq[0, FixedWing.DE] =   (- Cm0 - Cma * alpha) / Cmde
        de = u_eq[0, FixedWing.DE]

        det = Clda * Cndr - Cldr * Cnda


        u_eq[0, FixedWing.DA] = - Clb * beta * Cndr / det + Cnb * beta * Cldr / det
        da = u_eq[0, FixedWing.DA]

        u_eq[0, FixedWing.DR] =  Clb * beta * Cnda / det - Cnb * beta * Clda / det
        dr = u_eq[0, FixedWing.DR]

        FX = (Cx0 + Cxq * (q * bar_c / 2 / V)) * bar_p

        FY = - (Cyb * beta + b / 2 / V * (Cyp * p + Cyr * r) + Cydr * dr) * bar_p

        FZ = (Cz0 + Cza * alpha + q * bar_c / 2 / V * Czq + Czde * de) * bar_p

        # print(Cnb * beta + Cnda * da + Cndr * dr)

        # print("366")

        # print(V)

        Fbf = torch.tensor([FX, FY, FZ])

        Fbf = Fbf.detach().numpy()
        Fbf = np.array(Fbf).reshape(3,1)

        Rwb = torch.tensor([[c_a * c_b, s_b , s_a * c_b], [-c_a * s_b, c_b , -s_a * s_b], [-s_a, 0.0, c_a]])
        Rwb = Rwb.detach().numpy()
        Rwb = np.array(Rwb).reshape(3,3)

        Fwf = np.dot(Rwb, Fbf)

        D = Fwf[0]

        print(D)


        u_eq[:, FixedWing.T] =  m * grav * s_g + D / c_a / c_b


        return u_eq

    @property
    def u_eq(self):
        x = self.x
        params = self.params
        grav = params["g"]
        m = params["m"]

        rho = params["rho"]
        S = params["S"]
        bar_c = params["bar_c"]
        b = params["b"]

        CCd0 = params["Cd0"]
        Cda = params["Cda"]

        Clb = params["Clb"]
        Clda = params["Clda"]
        Cldr = params["Cldr"]
        Clp = params["Clp"]
        Clr = params["Clr"]

        Cm0 = params["Cm0"]
        Cma = params["Cma"]
        Cmde = params["Cmde"]
        Cmq = params["Cmq"]

        Cnb = params["Cnb"]
        Cndr = params["Cndr"]
        Cnda = params["Cnda"]
        Cnp = params["Cnp"]
        Cnr = params["Cnr"]

        Cyb = params["Cyb"]
        Cydr = params["Cydr"]
        Cyp = params["Cyp"]
        Cyr = params["Cyr"]

        Cz0 = params["Cz0"]
        Cza = params["Cza"]
        Czde = params["Czde"]
        Czq = params["Czq"]  

        Cx0 = params["Cx0"]
        Cxq = params["Cxq"]


        # print(x)
        # x = np.array(x)
        # x = torch.from_numpy(x)

        # V = x[:,FixedWing.V]
        V = x[:,FixedWing.V]
        alpha = x[:,FixedWing.ALPHA]
        beta = x[:,FixedWing.BETA]
        phi = x[:,FixedWing.PHI]
        gamma = x[:,FixedWing.GAMMA]
        psi = x[:,FixedWing.PSI]
        p = x[:,FixedWing.P]
        q = x[:,FixedWing.Q]
        r = x[:,FixedWing.R]

        bar_p = 1.0 / 2.0 * rho * rho * V * V * S



        
        s_a = torch.sin(alpha)
        c_a = torch.cos(alpha)

        s_b = torch.sin(beta)
        c_b = torch.cos(beta)
        t_b = torch.tan(beta)

        s_g = torch.sin(gamma)
        c_g = torch.cos(gamma)
        t_g = torch.tan(gamma)

        s_p = torch.sin(phi)
        c_p = torch.cos(phi)

        FX = (Cx0 + Cxq * (q * bar_c / 2 / V)) * bar_p

        FY = - (Cyb * beta + b / 2 / V * (Cyp * p + Cyr * r)) * bar_p

        FZ = (Cz0 + Cza * alpha + q * bar_c / 2 / V * Czq) * bar_p

        # print("366")

        # print(V)

        Fbf = torch.tensor([FX, FY, FZ])

        Fbf = Fbf.detach().numpy()
        Fbf = np.array(Fbf).reshape(3,1)

        Rwb = torch.tensor([[c_a * c_b, s_b , s_a * c_b], [-c_a * s_b, c_b , -s_a * s_b], [-s_a, 0.0, c_a]])
        Rwb = Rwb.detach().numpy()
        Rwb = np.array(Rwb).reshape(3,3)

        Fwf = np.dot(Rwb, Fbf)

        D = Fwf[0]

        Y = Fwf[1]

        L = Fwf[2]

        Fwx = torch.tensor([- D])
        Fwy = torch.tensor([- Y])
        Fwz = torch.tensor([- L])

        u_eq = torch.zeros((1, self.n_controls))
        # print(u_eq)
        u_eq[:, FixedWing.T] =  grav * s_g * m + D
        u_eq[0, FixedWing.DE] =   (- Cm0 - Cma * alpha) / Cmde
        det = Clda * Cndr - Cldr * Cnda
        u_eq[0, FixedWing.DA] = - Clb * beta * Cndr / det + Cnb * beta * Cldr / det
        u_eq[0, FixedWing.DR] =  Clb * beta * Cnda / det - Cnb * beta * Clda / det

        return u_eq