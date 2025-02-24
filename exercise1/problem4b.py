""" 
Course: FYS.420 Computational Physics
Name: Quy Le <quy.le@tuni.fi>
Student Id: 153118962
---------------------
Exercise 1 - Problem 4b: Numerical integration

This script implements Monte Carlo integration technique for approximating definite integrals.
In addition, the script also contains test functions to ensure the function works well.
"""
import numpy as np

def monte_carlo_integration(fun,xmin,xmax,blocks=10,iters=100):
    """
    Approximate the integral of a function using the Monte Carlo integration.

    Parameters:
    fun (function): The function to be integrated
    xmin (float): Lower limit of integration range
    xmax (float): Upper limit of integration range
    blocks (int): Number of sub-samples
    iters (int): Number of random samples per block
    """
    block_values = np.zeros((blocks,)) # Array storing result of blocks
    L = xmax-xmin # Width of integration range

    # Loop through each block
    for block in range(blocks):
        # For each blocks, samples #iters times
        for i in range(iters):
            # Get x_i randomly by normal random distribution
            x = xmin+np.random.rand()*L
            block_values[block]+=fun(x)
        # Get the mean of x_i values
        block_values[block]/=iters
    
    # Get the mean of block average
    I = L*np.mean(block_values)
    # Get the uncertainty of the result
    dI = L*np.std(block_values)/np.sqrt(blocks)
    return I,dI

def func(x):
    return np.sin(x)

def func2(x):
    return x**2

def main():
    I,dI = monte_carlo_integration(func,0.,np.pi/2,10,100)
    print("Integrated value: {0:0.5f} +/- {1:0.5f}".format(I,2*dI))

def test_monte_carlo_integration1():
    I1, dI = monte_carlo_integration(func, 0, 1)
    I_exact = -np.cos(1) + np.cos(0)
    assert abs(I1-I_exact)<3*dI

def test_monte_carlo_integration2():
    I1, dI = monte_carlo_integration(func, np.pi, 2*np.pi)
    I_exact = -np.cos(2*np.pi) + np.cos(np.pi)
    assert abs(I1-I_exact)<3*dI

def test_monte_carlo_integration3():
    I1, dI = monte_carlo_integration(func2, 0, 1)
    I_exact = 1/3
    assert abs(I1-I_exact)<3*dI


if __name__=="__main__":
    main()