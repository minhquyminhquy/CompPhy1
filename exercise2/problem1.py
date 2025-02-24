""" 
Course: FYS.420 Computational Physics 1
Name: Quy Le <quy.le@tuni.fi>
---------------------
Exercise 2 - Problem 1: Integration

This script calculate different integral in 1D, 2D, and 3D.
"""

import numpy as np
from scipy.integrate import quad, dblquad, tplquad, simpson, trapezoid
from scipy.integrate import nquad

# Part (a): One-dimensional integration

def part_a():
    f1 = lambda r: r**2 * np.exp(-2 * r)  
    f2 = lambda x: np.sin(x) / x 
    f3 = lambda x: np.exp(np.sin(x**3))  

    dx_s = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

    print("1a: integral 1")
    print("trapezoidal  simpson      dx")
    for dx in dx_s:
        x = np.arange(0, 20, dx)  
        y = f1(x)
        trapz_result = trapezoid(y, x)
        simpson_result = simpson(y, x=x)
        print(f"{trapz_result:.10f} {simpson_result:.10f} {dx:.10f}")

    integral_quad, error = quad(f1, 0, np.inf)
    print(f"quad = {integral_quad:.10f} +/- {error:.10f}")

    print("\n1a: integral 2")
    print("trapezoidal  simpson      dx")
    for dx in dx_s:
        x = np.arange(0.00001, 1, dx)  
        y = f2(x)
        trapz_result = trapezoid(y, x)
        simpson_result = simpson(y, x=x)
        print(f"{trapz_result:.10f} {simpson_result:.10f} {dx:.10f}")

    integral_quad, error = quad(f2, 0, 1)
    print(f"quad = {integral_quad:.10f} +/- {error:.10f}")

    print("\n1a: integral 3")
    print("trapezoidal  simpson      dx")
    for dx in dx_s:
        x = np.arange(0, 5, dx)  
        y = f3(x)
        trapz_result = trapezoid(y, x)
        simpson_result = simpson(y, x=x)
        print(f"{trapz_result:.10f} {simpson_result:.10f} {dx:.10f}")

    integral_quad, error = quad(f3, 0, 5)
    print(f"quad = {integral_quad:.10f} +/- {error:.10f}")


# Part (b): Two-dimensional integration


def part_b():
    # Function definition
    f = lambda x, y: (np.abs(x+y)) * np.exp(-0.5 * np.sqrt(x**2 + y**2))

    dy_s = [1e-1, 1e-2,1e-3,5e-4,2e-4]
    dx_s = [1e-1, 1e-2,1e-3,5e-4,2e-4]
    print("1b: two-dimensional integration")
    print("trapezoidal  simpson      dx*dy")
    
    for i in range(5):
        x = np.arange(0, 2, dx_s[i])
        y = np.arange(-2, 2, dy_s[i])
        X, Y = np.meshgrid(x, y)
        Z = f(X, Y)
        
        trapz_result = trapezoid(trapezoid(Z, y, axis=0), x)
        simpson_result = simpson(y=simpson(Z, x=y, axis=0), x=x)
  
        print(f"{trapz_result:.10f} {simpson_result:.10f} {dx_s[i]*dy_s[i]:.10f}")
    
    integral_nquad, error = nquad(f, [[0,2],[-2,2]])
    print(f"nquad = {integral_nquad:.10f} +/- {error:.10f}")

# Part (c): Three-dimensional integration

def psi_sqrt(r):
    rA = np.array([-1.1, 0.0, 0.0])[:, np.newaxis, np.newaxis, np.newaxis]
    rB = np.array([1.1, 0.0, 0.0])[:, np.newaxis, np.newaxis, np.newaxis]
    coeff = 1 / (2 * np.sqrt(3*np.pi))
    term1 = np.exp(-np.linalg.norm(r - rA, axis=0)**2 / 3)
    term2 = np.exp(-np.linalg.norm(r - rB, axis=0)**2 / 3)
    return np.abs(coeff * (term1 + term2))**2


def part_c():
    dx_s = [1.0, 0.5, 0.1]
    dy_s = [1.0, 0.5, 0.1]
    dz_s = [1.0, 0.5, 0.1]
    print("1c: three-dimensional integration")
    print("trapezoidal  simpson      dx*dy*dz")
    
    for dx,dy,dz in dx_s,dy_s,dz_s:
        x = np.arange(-10, 10, dx)
        y = np.arange(-10, 10, dy)
        z = np.arange(-10, 10, dz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        R = np.stack((X, Y, Z), axis=0)
        values = psi_sqrt(R)
        
        trapz_result = trapezoid(trapezoid(trapezoid(values, z, axis=2), y, axis=1), x)
        simpson_result = simpson(simpson(simpson(values, x=z, axis=2), x=y, axis=1), x=x)
        print(f"{trapz_result:.10f} {simpson_result:.10f} {dx**3:.10f}")

    def integrand(x, y, z):
        rA = np.array([-1.1, 0.0, 0.0])
        rB = np.array([1.1, 0.0, 0.0])
        term1 = np.exp(-np.sum((np.array([x, y, z]) - rA)**2) / 3)
        term2 = np.exp(-np.sum((np.array([x, y, z]) - rB)**2) / 3)
        psi = (term1 + term2) / (2 * np.sqrt(3 * np.pi))
        return psi**2

    bounds = [[-10, 10], [-10, 10], [-10, 10]]
    integral_nquad, error = nquad(integrand, bounds)

    print(f"nquad = {integral_nquad:.10f} +/- {error:.10f}")

if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
