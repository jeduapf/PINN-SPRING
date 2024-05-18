import numpy as np 
import torch

def forced_damped_spring(t, system_params, eps = 10**-9):
    # System parameters
    m = system_params["m"]
    b = system_params["b"]
    k = system_params["k"]

    # Forced input parameters
    f0 = system_params["F0"]
    ffreq = system_params["W"]
    x0 = system_params["x0"]
    x_0 = system_params["x_0"]

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

def exact_solution(d, w0, t):
    "Defines the analytical solution to the under-damped harmonic oscillator problem above."

    assert d < w0
    w = np.sqrt(w0**2-d**2)
    phi = np.arctan(-d/w)
    A = 1/(2*np.cos(phi))
    cos = torch.cos(phi+w*t)
    exp = torch.exp(-d*t)
    u = exp*2*A*cos

    return u
