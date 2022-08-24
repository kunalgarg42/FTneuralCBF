import numpy as np
from sympy.solvers import nsolve
from sympy import Symbol, sin, cos
import matplotlib.pyplot as plt
from dynamics.fixed_wing_dyn import fw_dyn
from CBF import CBF
from qp_control.qp_con import QP_con
import math

x0 = [10,0.05,0.08,0.01]

m = 23;

dt = 0.005
cla = 2*math.pi;
Cd0 = 0.02;
k1 = 0.05;
k2 = 0.05;

cm0 = -0.1;
cma = -0.1;
cmq = -9;
cmd = -1;
I = 4.51;
grav = 9.8;

rho = 1.225;
S = 0.99;
c = 0.33;

F = Symbol('F')
a = Symbol('a')

Vr = 25
gr = 0

f1 = -1/2*rho*(Vr**2)*S*(Cd0+cla*a+(cla**2)*(a**2))+F*cos(a)-m*grav*sin(gr)

f2 = 1/2*rho*(Vr**2)*S*(cla*a)+F*sin(a)-m*grav*cos(gr)

Ftr, alphar = nsolve((f1, f2),(F,a),(0,0))

print(alphar)
# B = CBF
# V = CBF
state1 =[]
state2 = []
state3 = []

def main():
	x = x0

	for i in range(20000):
		print(i)
		# x = x0
		fx, gx = fw_dyn(x)

		fx = np.reshape(fx, (4, 1))
		gx = np.reshape(gx, (4, 2))

		xr = [float(alphar), float(Ftr),Vr,gr]
		u = QP_con(x,xr,fx,gx)
		dx = fx.reshape(4,1) + np.matmul(gx,u).reshape(4,1)
		# dx = float(dx)
		dx = dx.reshape(4,1)
		dx = np.array(dx, dtype = float)
		# print(dx)
		x = np.array(x).reshape(4,1) + dx*dt
		# print(x)
		state1.append(x[0]-Vr)
		state2.append(x[1]-gr)
		state3.append(x[2]-gr-float(alphar))
	print(dx)

	# print(state1)
	fig, axs = plt.subplots(3)
	axs[0].plot(state1)
	axs[1].plot(state2)
	axs[2].plot(state3)
	plt.show()

if __name__ == '__main__':
	main()