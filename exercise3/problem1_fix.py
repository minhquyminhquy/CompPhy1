from scipy.integrate import simpson, nquad
import numpy as np

def simpson(f,x):
    """
    Approximate the integral of a function using Simpson's rule.

    Parameters:
    f (array): Function values at the given points.
    x (array): The x-values corresponding to the function values.

    Returns:
    float: The approximate integral value.
    """
    N = len(x)
    h = x[1] - x[0] # Width of the interval 

    # Calculate the sum for even and odd indexed terms based on Simpson's rule
    I_even = np.sum([f[:-2:2], 4*f[1:-1:2], f[2::2]])
    I_even = h/3 * I_even

    # In case of odd intervals, seperate and add up last slice
    if N % 2 == 0:
        delta_I = h/12 * (-f[N-3] + 8*f[N-2] + 5*f[N-1])
        I_even += delta_I

    return I_even 

def simpson3d(f,x,y,z):
    Idx = 0. * x
    for i in range(len(x)):
        Idy = 0. * y
        for j in range(len(y)):
            Idy[j]=simpson(f(x[i],y[j], z), x= z)
        Idx[j] = simpson(Idy, x=y)
    return simpson(Idx, x=x)

def func(x,y,z):
    """The example function to be integrate"""
    coeff = (x+y)**2
    exp = np.exp(-np.sqrt(x**2+y**2+z**2))
    sine = np.sin(z*np.pi)
    return coeff*exp*sine
x0, x1 = 0, 2
nXs = [8,12,16,20]
nX = nXs[-1]
nY = 2 * nX
nZ = int(3/2 * nX)
x = np.linspace(x0,x1,nX)
y = np.linspace(-2,2,nY)
z = np.linspace(-1,2,nZ)
print(simpson3d(func, x, y, z))