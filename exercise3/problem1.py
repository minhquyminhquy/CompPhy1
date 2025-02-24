""" 
Course: FYS.420 Computational Physics 1
Name: Quy Le <quy.le@tuni.fi>
---------------------
Exercise 3 - Problem 1: 3D integral revisited

This script calculate integral in 3D using Simpson in 2 ways: using meshgrid or the for-loop.
"""

from scipy.integrate import simpson, nquad
import numpy as np


def simpson_meshgrid(func, *args):
    """
    Calculates a 3D integral using Simpson rule with meshgrid.

    Parameters:
    func (function): Function to integrate
    args (array): Meshgrid array of x, y, z coordinate

    Returns:
    floats: the value of 3D integration
    """
    x,y,z = args[0],args[1],args[2]
    X,Y,Z = np.meshgrid(x,y,z,indexing="ij")
    values = func(X,Y,Z)

    integral_x = simpson(values, x=x, axis=0)
    integral_y = simpson(integral_x, x=y, axis=0)
    integral_z = simpson(integral_y, x=z, axis=0)

    return integral_z

def simpson_for_loop(func, *args):
    """
    Calculates a 3D integral using Simpson rule with for loops with accumulation.

    Parameters:
    func (function): Function to integrate
    args (array): Meshgrid array of x, y, z coordinate

    Returns:
    floats: the value of 3D integration
    """
    x,y,z = args[0],args[1],args[2]
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    dz = z[1]-z[0]
    nx = len(x)
    ny = len(y)
    nz = len(z)

    integral = 0.0
    for i in range(nz):
        for j in range(ny):
            for k in range(nx):
                if k % 2 == 1:
                    weight_x = 2
                elif (k != nx-1 and k !=0):
                    weight_x = 4
                else:
                    weight_x = 1
                
                if j % 2 == 1:
                    weight_y = 2
                elif (j != ny-1 and j !=0):
                    weight_y = 4
                else:
                    weight_y = 1

                if i % 2 == 1:
                    weight_z = 2
                elif (i != nz-1 and i !=0):
                    weight_z = 4
                else:
                    weight_z = 1

                # accumulate the integral
                integral += weight_x * weight_y * weight_z * func(x[k], y[j], z[i])

    integral = integral * (dx/3) * (dy/3) * (dz/3)
    return integral

def test_simpson_meshgrid():
    """Test the simpson_meshgrid integration method."""
    x0, x1 = 0, 2
    nX = 20
    nY = 2 * nX
    nZ = int(3 / 2 * nX)
    
    x = np.linspace(x0, x1, nX)
    y = np.linspace(-2, 2, nY)
    z = np.linspace(-1, 2, nZ)

    integral_meshgrid = simpson_meshgrid(func, x, y, z)
    bounds = [[x0, x1], [-2, 2], [-1, 2]]
    integral_nquad, error = nquad(func, bounds)

    tolerance = 1e-3  
    assert abs(integral_meshgrid - integral_nquad) < tolerance, \
        f"Test failed: Meshgrid integral {integral_meshgrid:.10f} != nquad result {integral_nquad:.10f}"

    print(f"test_simpson_meshgrid: Passed with value {integral_meshgrid:.10f}")


def test_simpson_for_loop():
    """Test the simpson_for_loop integration method."""
    x0, x1 = 0, 2
    nX = 20
    nY = 2 * nX
    nZ = int(3/2 * nX)
    
    x = np.linspace(x0, x1, nX)
    y = np.linspace(-2, 2, nY)
    z = np.linspace(-1, 2, nZ)

    integral_for_loop = simpson_for_loop(func, x, y, z)

    bounds = [[x0, x1], [-2, 2], [-1, 2]]
    integral_nquad, error = nquad(func, bounds)

    tolerance = 1e-1
    assert abs(integral_for_loop - integral_nquad) < tolerance, \
        f"Test failed: For-loop integral {integral_for_loop:.10f} != nquad result {integral_nquad:.10f}"

    print(f"test_simpson_for_loop: Passed with value {integral_for_loop:.10f}")

def func(x,y,z):
    """The example function to be integrate"""
    coeff = (x+y)**2
    exp = np.exp(-np.sqrt(x**2+y**2+z**2))
    sine = np.sin(z*np.pi)
    return coeff*exp*sine

def main():
    x0, x1 = 0, 2
    nXs = [8,12,16,20]
    nX = nXs[-1]
    nY = 2 * nX
    nZ = int(3/2 * nX)
    x = np.linspace(x0,x1,nX)
    y = np.linspace(-2,2,nY)
    z = np.linspace(-1,2,nZ)

    print("meshgrid      for_loop      dx*dy*dz")
    for nX in nXs:
        nY = 2 * nX
        nZ = int(3/2 * nX)
        x = np.linspace(x0,x1,nX)
        y = np.linspace(-2,2,nY)
        z = np.linspace(-1,2,nZ)

        dx = x[1]-x[0]
        dy = y[1]-y[0]
        dz = z[1]-z[0]

        integral_meshgrid = simpson_meshgrid(func, x, y, z)
        integral_for_loop = simpson_for_loop(func, x, y, z)

        print(f"{integral_meshgrid:.10f} {integral_for_loop:.10f} {dx*dy*dz:.10f}")

    bounds = [[x0, x1], [-2, 2], [-1, 2]]
    integral_nquad, error = nquad(func, bounds)
    print(f"nquad = {integral_nquad:.10f} +/- {error:.10f}")

if __name__ == "__main__":
    main()