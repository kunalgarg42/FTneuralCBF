import os
import sys
import torch
import numpy as np
import argparse
import random

sys.path.insert(1, os.path.abspath('..'))
sys.path.insert(1, os.path.abspath('.'))

from dynamics.Crazyflie import CrazyFlies
from trainer.constraints_crazy import constraints
from trainer.utils import Utils
from trainer.NNfuncgrad_CF import CBF

xg = torch.ones(1, 12)


x0 = 2 * torch.ones(1, 12)


dt = 0.01
n_state = 12
m_control = 4
fault = 0

nominal_params = {
    "m": 0.0299,
    "Ixx": 1.395 * 10**(-5),
    "Iyy": 1.395 * 10**(-5),
    "Izz": 2.173 * 10**(-5),
    "CT": 3.1582 * 10**(-10),
    "CD": 7.9379 * 10**(-12),
    "d": 0.03973,
    "fault": fault,}

state = []
goal = []


fault_control_index = 0
fault_duration = 0

fault_time = 0

def main():
	dynamics = CrazyFlies(x=x0, goal=xg, nominal_params=nominal_params, dt=dt)

	NN_cbf = CBF(dynamics, n_state=9, m_control=4)
	
	NN_cbf.load_state_dict(torch.load('./good_data/data/CF_cbf_NN_weightsCBF.pth'))

	NN_cbf.eval()

	dummy_input = torch.ones(1,12)

	V, JV = NN_cbf.V_with_jacobian(dummy_input)
	print(100 * JV.float())

	NN_cbf_trace = torch.jit.trace(NN_cbf,dummy_input)

	# print(NN_cbf_trace.code)

	NN_cbf_trace.save("./torch2c/traced_CBF_model_CF.pt")
	
	print(NN_cbf_trace(dummy_input))
		

if __name__ == '__main__':
	main()
