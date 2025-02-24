from scipy.integrate import simpson
import numpy as np

pi = np.pi

def simpson3D(fun, *args):
    
    pass

def fun(x,y,z):
    r = [x,y,z] # fxi
    rA = np.array([-1.1,0.,0.])
    rB = np.array([1.1,0.,0.])
    psi = (1/(2*np.sqrt(3*pi)))*[np.exp(-(r-rA))]
    psi_sqrt = psi**2

def a(a):
    pass