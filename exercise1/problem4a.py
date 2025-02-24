""" 
Course: FYS.420 Computational Physics
Name: Quy Le <quy.le@tuni.fi>
Student Id: 153118962
---------------------
Exercise 1 - Problem 4a: Numerical integration

This script implements three different numerical integration techniques for approximating definite integrals:
1. Riemann Sum
2. Trapezoidal Rule
3. Simpson's Rule

These methods are applied to a test function (sin(x)) over a specified interval [0, pi/2].
"""

import numpy as np

def riemann(f,x):
    """
    Approximate the integral of a function using the Riemann sum.

    Parameters:
    f (array): Function values at the given points.
    x (array): The x-values corresponding to the function values.

    Returns:
    I (float): The approximate integral value.
    """
    N = len(x) 
    I = 0 

    # Loop through the points to calcualte the sum of rectangle areas
    for i in range(N-1):
        delta_x_i = x[i+1] - x[i]
        f_x_i = f[i]
        I += delta_x_i * f_x_i

    return I

def trapezoid(f,x):
    """
    Approximate the integral of a function using the Trapezoidal rule.

    Parameters:
    f (array): Function values at the given points.
    x (array): The x-values corresponding to the function values.

    Returns:
    I (float): The approximate integral value.
    """
    N = len(x)
    I = 0

    # Loop through the intervals and compute the trapezoidal areas
    for i in range(N-1):
        delta_x_i = x[i+1] - x[i]
        f_sum = f[i] + f[i+1]
        I += delta_x_i * f_sum

    I /= 2
    return I

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

def test_riemann():
    # Define the test function f(x) = sin(x) and the interval [0, pi/2]
    n_intervals = 5
    upper_limit = np.pi/2
    lower_limit = 0
    x = np.linspace(lower_limit,upper_limit,n_intervals+1)
    f = np.sin(x)
    h = x[1] - x[0] # Width of interval
    epsilon = h # Tolerance
    
    # Calculate the approximation
    I_approx = riemann(f,x)
    I_exact = 1.0

    assert abs(I_approx-I_exact) < epsilon

def test_trapezoidal():
    # Define the test function f(x) = sin(x) and the interval [0, pi/2]
    n_intervals = 5
    upper_limit = np.pi/2
    lower_limit = 0
    x = np.linspace(lower_limit,upper_limit,n_intervals+1)
    f = np.sin(x)
    h = x[1] - x[0] # Width of interval
    epsilon = h # Tolerance

    # Calculate the approximation
    I_approx = riemann(f,x)
    I_exact = 1.0

    assert abs(I_approx-I_exact) < epsilon

def test_simpson():
    # Define the test function f(x) = sin(x) and the interval [0, pi/2]
    n_intervals = 5
    upper_limit = np.pi/2
    lower_limit = 0
    x = np.linspace(lower_limit,upper_limit,n_intervals+1)
    f = np.sin(x)
    h = x[1] - x[0] # Width of interval
    epsilon = h # Tolerance

    # Calculate the approximation
    I_approx = riemann(f,x)
    I_exact = 1.0

    assert abs(I_approx-I_exact) < epsilon

def main():
    # Define the test function f(x) = sin(x) and the interval [0, pi/2]
    x = np.linspace(0,np.pi/2,5)
    f = np.sin(x)
    print("The function f(x) = sin(x), x interval is [0, 2pi]")

    # Approximate the integral using the three methods 
    Ir = riemann(f,x)
    It = trapezoid(f,x)
    Is = simpson(f,x)

    print(f"Riemann sum approximation: {Ir}")
    print(f"Trapezoidal rule approximation: {It}")
    print(f"Simpson's rule approximation: {Is}")

if __name__ == "__main__":
    main()