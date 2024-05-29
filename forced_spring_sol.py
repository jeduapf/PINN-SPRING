import numpy as np
import matplotlib.pyplot as plt

def plot_sol(t,sol):
	plt.figure()
	plt.plot(t,sol)
	plt.show()

def forced_damped_spring(t, m, b, k, f0, ffreq, x0 = 0, x_0 = 0, eps = 10**-8):

	ksi = b/2/np.sqrt(m*k)
	w0 = np.sqrt(k/m)

	# --------- Homogeneous solution ---------
	
	# Amortecimento critico ( equal real exponential solutions )
	if np.abs(ksi - 1) < eps :
		xh = np.exp(-w0*t)*( x0*np.ones(t.shape) + (x_0 + w0*x0)*t )
		f1 = w0
		f2 = w0 

	# Superamortecido ( real exponential solutions )
	elif ksi - 1 > eps :
		wd = w0*np.sqrt(ksi**2-1)
		xh = np.exp(-ksi*w0*t)* ( ((x_0 + ksi*x0*w0)/wd)*np.sinh(wd*t) + x0*np.cosh(wd*t)  )
		f1 = -ksi*w0 + wd
		f2 = -ksi*w0 - wd
	
	# Subamortecido ( compelx exponential solutions )
	elif ksi - 1 < eps :
		wd = w0*np.sqrt(1-ksi**2)
		xh = np.exp(-ksi*w0*t)* ( ((x_0 + ksi*x0*w0)/wd)*np.sin(wd*t) + x0*np.cos(wd*t)  )
		f1 = -ksi*w0 + wd
		f2 = -ksi*w0 - wd

	else:
		xh = None

	# --------- Forced sinousoidal solution ---------
	
	xf = ( f0/( (2*ksi*w0*ffreq)**2 + (w0**2 - ffreq**2)**2 ) )*( 2*ksi*w0*ffreq*np.sin(ffreq*t) + (w0**2 - ffreq**2)*np.cos(ffreq*t) )
	
	x = xh + xf

	return x.astype(float)

if __name__ == "__main__":

	Tf = 5
	N = 10**6
	t = np.linspace(0,Tf,N)
	m = 1.0
	b = 4.0
	k = 400.0
	ffreq = 15.0
	f0 = 20.0
	x0 = 1.0
	x_0 = 0.0

	sol = forced_damped_spring(t, m, b, k, f0, ffreq, x0, x_0, eps = 10**-8)

	plot_sol(t,sol)