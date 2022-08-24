import numpy as np
import math
# m = 1
# g = 9.8
# zeng = -0.1
# Cl = 0
# Cm = 0
# Cn = 0
# CL0 = 0
# CLa = 0
# Clq = 0
# CLd = 0

# CDo = 0
# CDcl = 0

# Cyb = 0
# Cyp = 0
# Cyr = 0
# Cyd = 0

m = 23

cla = 2*math.pi
Cd0 = 0.02
k1 = 0.05
k2 = 0.05

cm0 = -0.1
cma = -0.1
cmq = -9
cmd = -1
I = 4.51
grav = 9.8

rho = 1.225
S = 0.99
c = 0.33
gr = 0.1

fx = []
gx = []

def fw_dyn(x):
	alpha = x[2]-x[1]
	theta = x[2]
	v = x[0]
	q = x[3]
	gamma = x[1]
	Cl = cla*alpha
	Cd = Cd0 + k1*Cl + k2*Cl**2
	Cmf = cm0 + cma*alpha+cmq*q
	Cmg = cmd   
	L = 1/2*rho*(v**2)*S*Cl
	D = 1/2*rho*(v**2)*S*Cd   
	Mf = 1/2*rho*(v**2)*S*c*Cmf
	Mg = 1/2*rho*(v**2)*S*c*Cmg
	fx = [-1/m*D-grav*math.sin(gamma), 1/m/v*L-grav/v*math.cos(gamma), q,Mf/I]
	gx = [[1/m*math.cos(alpha), 0],[1/m*math.sin(alpha)/v,0], [0, 0], [0, Mg/I]]

	return np.array(fx,dtype = object), np.array(gx,dtype = object)

def fw_dyn_ext(state,u_in,N):
	# N = int(state.size)
	# print(state[0])
	u_in = u_in.detach().numpy()
	dxdt = []
	state = np.array(state,dtype = float)
	# print(state[N])
	# print(state.shape)
	for i in range(N):
		# print(i)
		x = np.array(state[i], dtype = float).reshape(4,1)
		u = np.array(u_in[i], dtype = float).reshape(2,1)
		alpha = x[2]-x[1]
		theta = x[2]
		v = x[0]
		q = x[3]
		gamma = x[1]
		gamma = np.mod(gamma,2*math.pi)
		# print(gamma)
		alpha = np.mod(alpha,2*math.pi)
		Cl = cla*alpha
		Cd = Cd0 + k1*Cl + k2*Cl**2
		Cmf = cm0 + cma*alpha+cmq*q
		Cmg = cmd   
		L = 1/2*rho*(v**2)*S*Cl
		D = 1/2*rho*(v**2)*S*Cd   
		Mf = 1/2*rho*(v**2)*S*c*Cmf
		Mg = 1/2*rho*(v**2)*S*c*Cmg
		fx = [-1/m*D-grav*math.sin(gamma), 1/m/v*L-grav/v*math.cos(gamma), q,Mf/I]
		gx = [[1/m*math.cos(alpha), 0],[1/m*math.sin(alpha)/v,0], [0, 0], [0, Mg/I]]

		fx = np.array(fx,dtype = object)
		gx = np.array(gx,dtype = object)

		dx = np.array(fx,dtype = float).reshape(4,1) + np.matmul(gx,u).reshape(4,1)
		# dx = float(dx)
		dx = dx.reshape(4,1)
		dxdt.append(dx)
	return dxdt
	# return np.array(fx,dtype = float), np.array(gx,dtype = float)

