TRAIN_STEPS = 100000000
EVAL_STEPS = 2000 #  10000
EVAL_EPOCHS = 100
POLICY_UPDATE_INTERVAL = 20000
INIT_STATE_UPDATE = 500
FAULT_DURATION = EVAL_STEPS

TRAJ_LEN = 100

fault = 0  # int(input("Fault (1) or pre-fault (0): "))

FIXED_WING_PARAMS = {
    "m": 100.0,
    "g": 9.8,
    "Ixx": 100,
    "Iyy": 1000,
    "Izz": 1000,
    "Ixz": 100,
    "S": 100,
    "b": 5,
    "bar_c": 5,
    "rho": 1.2,
    "Cd0": 0.0434,
    "Cda": 0.22,
    "Clb": -0.13,
    "Clp": -0.505,
    "Clr": 0.252,
    "Clda": 0.0855,
    "Cldr": -0.0024,
    "Cm0": 0.135,
    "Cma": -1.50,
    "Cmq": -38.2,
    "Cmde": -0.992,
    "Cnb": 0.0726,
    "Cnp": -0.069,
    "Cnr": -0.0946,
    "Cnda": 0.7,
    "Cndr": -0.0693,
    "Cyb": -0.83,
    "Cyp": 0,
    "Cyr": 0,
    "Cydr": 0.1,
    "Cz0": 0.23,
    "Cza": 4.58,
    "Czq": 0,
    "Czde": 0.1,
    "Cx0": 0,
    "Cxq": 0,
    "fault": fault, }


CRAZYFLIE_PARAMS = {
    "m": 0.0299,
    "Ixx": 1.395 * 10**(-5),
    "Iyy": 1.395 * 10**(-5),
    "Izz": 2.173 * 10**(-5),
    "CT": 3.1582 * 10**(-10),
    "CD": 7.9379 * 10**(-12),
    "d": 0.03973,
    "fault": fault,}

CRAZYFLIE_PARAMS_PERT = {
    "m": 0.015,
    "Ixx": 1 * 10**(-5),
    "Iyy": 1 * 10**(-5),
    "Izz": 1 * 10**(-5),
    "CT": 2 * 10**(-10),
    "CD": 5 * 10**(-12),
    "d": 0.02,
    "fault": fault,}

CRAZYFLIE_PARAMS_PERT_2 = {
    "m": 0.02,
    "Ixx": 2 * 10**(-5),
    "Iyy": 1 * 10**(-5),
    "Izz": 3 * 10**(-5),
    "CT": 2.5 * 10**(-10),
    "CD": 9 * 10**(-12),
    "d": 0.05,
    "fault": fault,}